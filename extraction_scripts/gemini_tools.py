import os
import glob
import re
import csv
import json
import sys
from collections import Counter
from typing import List, Dict, Set, Any
from openai import OpenAI

# =========================================
# CONFIGURATION
# =========================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 
HPO_JSON_FILE = "hp.json"
NUM_RUNS = 50

# If a run produces NO parseable HPO output (parser gets 0 terms) or errors,
# retry up to this many times to still end up with NUM_RUNS usable runs.
MAX_RETRIES_PER_RUN = 8

# Gemini 2.5 Pro is a native "Thinking" model.
# Thinking is always on and included in output tokens.
MODEL_NAME = "gemini-2.5-pro" 

# === COST TRACKING ===
# Standard Gemini 2.5 Pro Pricing
PRICE_INPUT_PER_1M = 1.25
PRICE_OUTPUT_PER_1M = 10.0
# =========================================


# =========================================
# HPO DATA LOADING (Standard)
# =========================================

HPO_DATA: List[Dict[str, Any]] = []
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

        for term in HPO_DATA:
            raw_id = term.get('id', '')
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
    """Tool function to search HPO terms."""
    if not HPO_DATA:
        return "Error: Database not loaded."

    query = query.lower().strip()
    matches = []

    for term in HPO_DATA:
        term_id = term.get('id', '')
        term_lbl = term.get('lbl', term.get('name', ''))

        meta = term.get('meta', {})
        synonyms_list = meta.get('synonyms', [])
        term_synonyms = []
        for s in synonyms_list:
            if isinstance(s, dict):
                term_synonyms.append(s.get('val', '').lower())
            elif isinstance(s, str):
                term_synonyms.append(s.lower())

        score = 100

        if query == term_id.lower():
            score = 1
        elif query == term_lbl.lower():
            score = 2
        elif query in term_synonyms:
            score = 3
        elif term_lbl.lower().startswith(query):
            score = 4
        elif query in term_lbl.lower() or query in term_id.lower():
            score = 5

        if score < 100:
            matches.append({
                "id": term_id,
                "name": term_lbl,
                "score": score
            })

    matches.sort(key=lambda x: (x['score'], len(x['name'])))
    return json.dumps(matches[:15], indent=2)


# =========================================
# GEMINI CLIENT
# =========================================

if GEMINI_API_KEY is None or GEMINI_API_KEY.startswith("YOUR_"):
    print("WARNING: Please set GEMINI_API_KEY.")

try:
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    sys.exit(1)


# =========================================
# PROMPT & TOOLS
# =========================================

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
                    "query": {
                        "type": "string",
                        "description": "The clinical feature to search for."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def extract_hpo_terms(text: str):
    pattern = r'-\s*\[(HP:\d+)\]\s*(.+)'
    matches = re.findall(pattern, text)
    return [(m[0], m[1].strip()) for m in matches]


def _usage_from_openai_compat(resp) -> dict:
    """Extract usage, attempting to catch reasoning tokens if exposed."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}

    out = {}
    # Standard fields
    for k in ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"]:
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out[k] = v

    # Try to grab reasoning tokens (OpenAI standard location)
    # Gemini might map them here in future updates
    details = getattr(usage, "completion_tokens_details", None)
    if details:
        out["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0)

    return out


def _estimate_cost_from_usage(usage: dict):
    in_toks = usage.get("input_tokens", usage.get("prompt_tokens"))
    out_toks = usage.get("output_tokens", usage.get("completion_tokens"))
    if not (isinstance(in_toks, int) and isinstance(out_toks, int)):
        return None
    return (in_toks / 1_000_000.0) * PRICE_INPUT_PER_1M + (out_toks / 1_000_000.0) * PRICE_OUTPUT_PER_1M


# =========================================
# MAIN BENCHMARK LOGIC
# =========================================

def process_file_reliability(filepath: str):
    print(f"\n=== Processing: {filepath} ({NUM_RUNS} runs - GEMINI WITH TOOLS) ===")

    with open(filepath, 'r', encoding='utf-8') as f:
        note_content = f.read()

    term_counter = Counter()
    raw_log_data = []
    cost_log = []

    run_num = 1
    while run_num <= NUM_RUNS:
        attempt = 1

        agg_in_total = 0
        agg_out_total = 0
        agg_tot_total = 0
        any_usage_known = False

        last_error = None
        last_final_text = ""

        while attempt <= MAX_RETRIES_PER_RUN:
            try:
                print(f"  > Run {run_num}/{NUM_RUNS} (attempt {attempt}/{MAX_RETRIES_PER_RUN})...", end="\r", flush=True)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": note_content},
                ]

                # ---- Initial call ----
                # NO temperature parameter = Default (Correct for Thinking Model)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS_DEFINITION,
                    tool_choice="auto"
                )

                usage1 = _usage_from_openai_compat(response)
                in1 = usage1.get("input_tokens", usage1.get("prompt_tokens"))
                out1 = usage1.get("output_tokens", usage1.get("completion_tokens"))
                tot1 = usage1.get("total_tokens", (in1 or 0) + (out1 or 0))

                msg = response.choices[0].message
                messages.append(msg)

                # ---- Handle tool calls ----
                final_text = ""
                in2 = out2 = tot2 = 0

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.function.name == "search_hpo_terms":
                            args = json.loads(tool_call.function.arguments)
                            result = search_hpo_terms(args.get("query", ""))

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            })

                    # ---- Final model call ----
                    # NO temperature parameter = Default (Correct for Thinking Model)
                    final_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                    )
                    final_text = final_response.choices[0].message.content or ""

                    usage2 = _usage_from_openai_compat(final_response)
                    in2 = usage2.get("input_tokens", usage2.get("prompt_tokens"))
                    out2 = usage2.get("output_tokens", usage2.get("completion_tokens"))
                    tot2 = usage2.get("total_tokens", (in2 or 0) + (out2 or 0))
                else:
                    final_text = msg.content or ""

                last_final_text = final_text or ""

                # ---- Aggregate Cost/Tokens ----
                in_total = (in1 or 0) + (in2 or 0)
                out_total = (out1 or 0) + (out2 or 0)
                tot_total = (tot1 or 0) + (tot2 or 0)

                if isinstance(in1, int) or isinstance(in2, int) or isinstance(out1, int) or isinstance(out2, int):
                    any_usage_known = True

                agg_in_total += in_total
                agg_out_total += out_total
                agg_tot_total += tot_total

                # ---- Extract HPO terms ----
                terms = extract_hpo_terms(final_text or "")

                # If parser yields nothing, retry this run (do not advance run_num yet)
                if not terms:
                    attempt += 1
                    continue

                # ---- Success: log exactly once for this run_num ----
                est_cost = None
                if any_usage_known and isinstance(PRICE_INPUT_PER_1M, (int, float)):
                    est_cost = (agg_in_total / 1_000_000.0) * PRICE_INPUT_PER_1M + (agg_out_total / 1_000_000.0) * PRICE_OUTPUT_PER_1M
                cost_log.append([run_num, agg_in_total, agg_out_total, agg_tot_total, est_cost])

                term_counter.update(terms)
                for hpo_id, term_name in terms:
                    is_valid = str(hpo_id in VALID_HPO_IDS)
                    raw_log_data.append([run_num, hpo_id, term_name, is_valid])

                break  # move to next run_num

            except Exception as e:
                last_error = e
                attempt += 1
                continue

        else:
            # Exceeded retries: record outcome for this run_num and move on
            if last_error is not None:
                print(f"\n  Error on run {run_num}: {last_error}")
                raw_log_data.append([run_num, "ERROR", str(last_error), "False"])
            else:
                raw_log_data.append([run_num, "NO_TERMS_FOUND", "N/A", "False"])

            if any_usage_known and isinstance(PRICE_INPUT_PER_1M, (int, float)):
                est_cost = (agg_in_total / 1_000_000.0) * PRICE_INPUT_PER_1M + (agg_out_total / 1_000_000.0) * PRICE_OUTPUT_PER_1M
                cost_log.append([run_num, agg_in_total, agg_out_total, agg_tot_total, est_cost])
            else:
                cost_log.append([run_num, None, None, None, None])

        run_num += 1

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # === Export 1: Summary ===
    output_summary = f"{base_name}_GEMINI_WITH_TOOLS_summary.csv"
    with open(output_summary, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HPO ID", "Term Name", "Count (out of 50)", "Percentage"])
        for (hpo_id, term_name), count in term_counter.most_common():
            percentage = (count / NUM_RUNS) * 100.0
            writer.writerow([hpo_id, term_name, count, f"{percentage:.1f}%"])

    # === Export 2: Raw Log ===
    output_raw = f"{base_name}_GEMINI_WITH_TOOLS_raw_log.csv"
    with open(output_raw, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "HPO ID", "Term Name", "Is Valid ID in DB?"])
        writer.writerows(raw_log_data)

    # === Export 3: Cost Log ===
    output_cost = f"{base_name}_GEMINI_WITH_TOOLS_cost_log.csv"
    with open(output_cost, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Run Number",
            "Input Tokens (Total)",
            "Output Tokens (Includes Thinking)",  # <--- UPDATED HEADER
            "Total Tokens",
            "Estimated Cost USD"
        ])
        writer.writerows(cost_log)

    print(f"\n  Saved summary to {output_summary}")
    print(f"  Saved raw log to {output_raw}")
    print(f"  Saved cost log to {output_cost}")


def main():
    load_data(HPO_JSON_FILE)
    if not HPO_DATA:
        print("No HPO data loaded. Exiting.")
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
