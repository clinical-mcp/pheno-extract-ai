import os
import glob
import re
import csv
import json
import sys
from collections import Counter
from typing import List, Dict, Set
from openai import OpenAI

# === CONFIGURATION ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HPO_JSON_FILE = "hp.json"
NUM_RUNS = 50
MODEL_NAME = "gpt-5.1-2025-11-13" # or "gpt-5.1" if you have access
# =====================

# If a run produces NO parseable HPO output (parser gets 0 terms) or errors,
# retry up to this many times before starting the run over again.
MAX_RETRIES_PER_RUN = 8
# =====================


# === COST TRACKING (optional $ estimates; tokens always logged if provided by API) ===
PRICE_INPUT_PER_1M = 1.25   # e.g., 5.0
PRICE_OUTPUT_PER_1M = 10    # e.g., 15.0
# =====================

# Global sets for fast validation
HPO_DATA: List[Dict[str, str]] = []
VALID_HPO_IDS: Set[str] = set()

def load_data(filepath: str = "hp.json"):
    """Loads HPO data and creates a set of valid IDs for validation."""
    global HPO_DATA, VALID_HPO_IDS
    try:
        print(f"Loading HPO data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'graphs' in data:
            HPO_DATA = data['graphs'][0]['nodes']
        elif isinstance(data, list):
            HPO_DATA = data
        else:
            HPO_DATA = data

        # Create a set of all valid IDs for fast checking
        # Assuming ID format in JSON is like "http://purl.obolibrary.org/obo/HP_0001234" or "HP:0001234"
        # We normalize to "HP:0001234"
        for term in HPO_DATA:
            raw_id = term.get('id', '')
            # Normalize ID if it looks like a URL
            if 'HP_' in raw_id:
                clean_id = 'HP:' + raw_id.split('HP_')[-1]
                VALID_HPO_IDS.add(clean_id)
            else:
                VALID_HPO_IDS.add(raw_id)

        print(f"SUCCESS: Loaded {len(HPO_DATA)} terms. Valid ID set created.")
    except Exception as e:
        print(f"ERROR loading JSON: {e}")
        sys.exit(1)

def search_hpo_terms(query: str) -> str:
    """
    Searches HPO terms with RANKING:
    1. Exact Name Match (e.g. "Seizures" == "Seizures")
    2. Exact Synonym Match
    3. Starts With (e.g. "Seizures" -> "Seizures, generalized")
    4. Substring Match
    """
    if not HPO_DATA:
        return "Error: Database not loaded."

    query = query.lower().strip()
    matches = []

    for term in HPO_DATA:
        term_id = term.get('id', '')
        term_lbl = term.get('lbl', term.get('name', ''))
        term_synonyms = [s.get('val', '') for s in term.get('meta', {}).get('synonyms', [])]

        # Calculate a "Score" (Lower is better)
        score = 100  # Default: No match

        # 1. Exact ID Match (Best)
        if query == term_id.lower():
            score = 1

        # 2. Exact Name Match
        elif query == term_lbl.lower():
            score = 2

        # 3. Exact Synonym Match
        elif query in [s.lower() for s in term_synonyms]:
            score = 3

        # 4. Starts With
        elif term_lbl.lower().startswith(query):
            score = 4

        # 5. Simple Substring (Weakest)
        elif query in term_lbl.lower() or query in term_id.lower():
            score = 5

        if score < 100:
            matches.append({
                "id": term_id,
                "name": term_lbl,
                "score": score
            })

    # SORT BY SCORE, then by Name Length (shorter is usually more generic/correct)
    matches.sort(key=lambda x: (x['score'], len(x['name'])))

    # Return top 15 highest quality matches
    return json.dumps(matches[:15], indent=2)

def _usage_from_resp(resp) -> dict:
    """
    Robust usage extraction including HIDDEN REASONING tokens.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}

    if isinstance(usage, dict):
        return usage

    out = {}
    for k in ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"]:
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out[k] = v

    # Capture Reasoning Tokens
    details = getattr(usage, "completion_tokens_details", None)
    if details:
        out["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0)
    else:
        out["reasoning_tokens"] = 0

    return out

def _estimate_cost(in_toks, out_toks):
    if not (isinstance(in_toks, int) and isinstance(out_toks, int)):
        return None
    if not (isinstance(PRICE_INPUT_PER_1M, (int, float)) and isinstance(PRICE_OUTPUT_PER_1M, (int, float))):
        return None
    return (in_toks / 1_000_000.0) * PRICE_INPUT_PER_1M + (out_toks / 1_000_000.0) * PRICE_OUTPUT_PER_1M

# === CLIENT ===
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing client: {e}")
    exit()

# === PROMPT & TOOLS ===
SYSTEM_PROMPT = """
You are an expert Clinical Geneticist and phenotype extractor.

Goal: Produce a COMPREHENSIVE but ACCURATE list of patient phenotypic abnormalities from this note, mapped to HPO.
Do NOT invent findings. Only include a term if it is clearly supported in the note (explicitly stated or confidently inferable from objective data).

Critical inclusion/exclusion rules:
- Count ONLY findings in the PATIENT. Family history does NOT count unless the note explicitly states the patient has the finding.
- Exclude negated findings (e.g., “no seizures”, “denies seizures”, “negative for …”).
- Exclude uncertainty (“possible”, “concern for”, “rule out”) unless later confirmed.
- Do NOT add “expected” features of a suspected diagnosis unless documented in the patient.
- Avoid duplicative near-synonyms; prefer the most appropriate HPO concept(s). You may include BOTH a high-level parent and a specific child term if both are supported and non-redundant.

Inference rules (high recall, controlled):
You MAY infer a phenotype ONLY when objective data meet common clinical thresholds or patterns in the note:
- Microcephaly: OFC/HC < 3rd percentile or z ≤ -2
- Macrocephaly: OFC/HC > 97th percentile or z ≥ +2
- Short stature: height < 3rd percentile or z ≤ -2
- Tall stature: height > 97th percentile or z ≥ +2
- Underweight/failure to thrive: weight < ~3rd–5th percentile or z ≤ -2, or clear longitudinal concern
- Anemia: low hemoglobin/hematocrit; specify subtype only if stated or clearly supported (e.g., “normocytic”)
- Hypothyroidism: elevated TSH with low free T4 (or explicit diagnosis)

Workflow (MUST follow):
1) Read the ENTIRE note.
2) Build an internal problem list of all abnormalities (include dysmorphology, growth, neurodevelopment, MSK, cardiac, heme, endocrine, etc.).
3) For each problem-list item, decide:
   - DIRECT (explicitly stated), or
   - INFERRED (supported by objective data meeting thresholds/patterns above).
   If not clearly supported, exclude it.
4) Use the `search_hpo_terms` tool ONLY for the interpreted phenotype concept (not raw numbers, treatments, devices).
   - BAD: `search_hpo_terms("TSH 14")`
   - BAD: `search_hpo_terms("CPAP")`
   - GOOD: `search_hpo_terms("Hypothyroidism")`
   - GOOD: `search_hpo_terms("Microcephaly")`
5) Final output: a clean list only.

Output format (ONLY):
- [HP:########] Official Term Name
"""

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_hpo_terms",
            "description": "Searches for HPO terms matching the query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The clinical feature to search for."}
                },
                "required": ["query"]
            }
        }
    }
]

def extract_hpo_terms(text):
    pattern = r'-\s*\[(HP:\d+)\]\s*(.+)'
    matches = re.findall(pattern, text)
    return [(m[0], m[1].strip()) for m in matches]

def process_file_reliability(filepath):
    print(f"\n=== Processing: {filepath} ({NUM_RUNS} runs - WITH TOOLS) ===")

    with open(filepath, 'r', encoding='utf-8') as f:
        note_content = f.read()

    term_counter = Counter()
    raw_log_data = []
    cost_log_data = []

    run_num = 1
    while run_num <= NUM_RUNS:
        attempt = 1
        last_error = None
        success = False

        while attempt <= MAX_RETRIES_PER_RUN:
            try:
                print(
                    f"  > Run {run_num}/{NUM_RUNS} (attempt {attempt}/{MAX_RETRIES_PER_RUN})...",
                    end='\r',
                    flush=True
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": note_content}
                ]

                # 1. Initial Call
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS_DEFINITION,
                    tool_choice="auto",
                    reasoning_effort="medium",
                )
                msg = response.choices[0].message
                messages.append(msg)

                # capture usage from first call
                usage1 = _usage_from_resp(response)
                in1 = usage1.get("input_tokens", usage1.get("prompt_tokens"))
                out1 = usage1.get("output_tokens", usage1.get("completion_tokens"))
                reas1 = usage1.get("reasoning_tokens", 0)

                tot1 = usage1.get("total_tokens")
                if tot1 is None and isinstance(in1, int) and isinstance(out1, int):
                    tot1 = in1 + out1

                # 2. Handle Tool Calls
                final_response = None
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.function.name == "search_hpo_terms":
                            args = json.loads(tool_call.function.arguments)
                            result = search_hpo_terms(args.get("query"))
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result
                            })

                    # 3. Final Answer
                    final_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        reasoning_effort="medium",
                    )
                    final_text = final_response.choices[0].message.content or ""
                else:
                    final_text = msg.content or ""

                # capture usage from final call (if it happened)
                usage2 = _usage_from_resp(final_response) if final_response is not None else {}
                in2 = usage2.get("input_tokens", usage2.get("prompt_tokens"))
                out2 = usage2.get("output_tokens", usage2.get("completion_tokens"))
                reas2 = usage2.get("reasoning_tokens", 0)

                tot2 = usage2.get("total_tokens")
                if tot2 is None and isinstance(in2, int) and isinstance(out2, int):
                    tot2 = in2 + out2

                def _sum(a, b):
                    if isinstance(a, int) and isinstance(b, int):
                        return a + b
                    if isinstance(a, int) and b is None:
                        return a
                    if a is None and isinstance(b, int):
                        return b
                    return None

                run_in = _sum(in1 if isinstance(in1, int) else None, in2 if isinstance(in2, int) else None)
                run_out = _sum(out1 if isinstance(out1, int) else None, out2 if isinstance(out2, int) else None)
                run_reas = _sum(reas1 if isinstance(reas1, int) else None, reas2 if isinstance(reas2, int) else None)
                run_total = _sum(tot1 if isinstance(tot1, int) else None, tot2 if isinstance(tot2, int) else None)

                if run_total is None and isinstance(run_in, int) and isinstance(run_out, int):
                    run_total = run_in + run_out

                est_cost = _estimate_cost(run_in, run_out)

                # 4. Extract
                terms = extract_hpo_terms(final_text)

                # If parser yields nothing, retry this run (do not count it)
                if not terms:
                    attempt += 1
                    continue

                # 5. Log ONLY the successful run (exactly once for this run_num)
                cost_log_data.append([run_num, run_in, run_out, run_reas, run_total, est_cost, bool(msg.tool_calls)])

                term_counter.update(terms)
                for hpo_id, term_name in terms:
                    is_valid = str(hpo_id in VALID_HPO_IDS)
                    raw_log_data.append([run_num, hpo_id, term_name, is_valid])

                success = True
                break

            except Exception as e:
                last_error = e
                attempt += 1
                continue

        # If still no parseable output after retries, do NOT advance run_num.
        # Start this same run over again so you end with NUM_RUNS successful runs.
        if not success:
            if last_error is not None:
                print(f"\n  Run {run_num} failed after {MAX_RETRIES_PER_RUN} attempts (last error: {last_error}). Retrying run {run_num}...")
            else:
                print(f"\n  Run {run_num} produced no parseable terms after {MAX_RETRIES_PER_RUN} attempts. Retrying run {run_num}...")
            continue

        run_num += 1

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # === Export 1: Aggregate Summary ===
    output_summary = f"{base_name}_GPT_WITH_TOOLS_summary.csv"
    with open(output_summary, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HPO ID", "Term Name", "Count (out of 50)", "Percentage"])
        for (hpo_id, term_name), count in term_counter.most_common():
            percentage = (count / NUM_RUNS) * 100
            writer.writerow([hpo_id, term_name, count, f"{percentage:.1f}%"])

    # === Export 2: Raw Run Log ===
    output_raw = f"{base_name}_GPT_WITH_TOOLS_raw_log.csv"
    with open(output_raw, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "HPO ID", "Term Name", "Is Valid ID in DB?"])
        writer.writerows(raw_log_data)

    # === Export 3: Token/Cost Log (UPDATED HEADERS) ===
    output_cost = f"{base_name}_GPT_WITH_TOOLS_cost_log.csv"
    with open(output_cost, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Run Number",
            "Input Tokens (sum)",
            "Output Tokens (sum)",
            "Reasoning Tokens (sum)",
            "Total Tokens (sum)",
            "Estimated Cost USD",
            "Used Tools (True/False)",
        ])
        writer.writerows(cost_log_data)

    print(f"\n  Saved summary to {output_summary}")
    print(f"  Saved raw log to {output_raw}")
    print(f"  Saved cost log to {output_cost}")

def main():
    load_data(HPO_JSON_FILE)
    if not HPO_DATA:
        return

    txt_files = glob.glob("*.txt")
    files = [
        f for f in txt_files
        if "summary" not in f and "raw_log" not in f and "hpo_terms" not in f and "cost_log" not in f
    ]

    if not files:
        print("No input .txt files found.")
        return

    for f in files:
        process_file_reliability(f)

if __name__ == "__main__":
    main()
