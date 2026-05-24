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
MODEL_NAME = "gpt-5.1-2025-11-13" # or "gpt-5.1"
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

# === CLIENT ===
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing client: {e}")
    exit()

# === PROMPT (NO TOOLS VERSION) ===
SYSTEM_PROMPT_NO_TOOLS = """
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
4) Assign the most appropriate official HPO ID for each included phenotype using your internal knowledge.
   If you are not confident of the exact HPO ID, EXCLUDE the term rather than guessing.
5) Final output: a clean list only.

Output format (ONLY):
- [HP:########] Official Term Name
"""

def extract_hpo_terms(text: str):
    """
    Extract HPO lines in either of these formats:
      - [HP:0001234] Term Name
        [HP:0001234] Term Name

    Returns a list of tuples: (ID, Name), deduped by ID (order preserved).
    """
    if not text:
        return []

    line_re = re.compile(r'^\s*(?:-\s*)?\[(HP:\d+)\]\s*(.+?)\s*$', re.MULTILINE)
    matches = line_re.findall(text)

    # Deduplicate by HPO ID, preserve order
    seen = set()
    out = []
    for hpo_id, name in matches:
        if hpo_id not in seen:
            seen.add(hpo_id)
            out.append((hpo_id, name.strip()))
    return out

def _usage_from_resp(resp) -> dict:
    """
    Robust extraction of usage, including hidden REASONING tokens.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}

    if isinstance(usage, dict):
        return usage

    out = {}
    # Standard tokens
    for k in ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"]:
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out[k] = v

    # Capture Reasoning Tokens (OpenAI puts these inside completion_tokens_details)
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

def process_file_reliability_no_tools(filepath):
    print(f"\n=== Processing: {filepath} ({NUM_RUNS} runs - NO TOOLS) ===")

    with open(filepath, 'r', encoding='utf-8') as f:
        note_content = f.read()

    term_counter = Counter()

    # List to store raw row data
    raw_log_data = []

    # per-run token/cost tracking (for the final accepted run)
    cost_log_data = []

    run_num = 1
    while run_num <= NUM_RUNS:
        attempt = 1
        last_error = None
        success = False

        while attempt <= MAX_RETRIES_PER_RUN:
            try:
                print(f"  > Run {run_num}/{NUM_RUNS} (attempt {attempt}/{MAX_RETRIES_PER_RUN})...", end='\r', flush=True)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_NO_TOOLS},
                    {"role": "user", "content": note_content}
                ]

                # === CALL WITHOUT TOOLS ===
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    reasoning_effort="medium"
                )

                final_text = response.choices[0].message.content or ""

                # capture usage including reasoning
                usage = _usage_from_resp(response)
                in_toks = usage.get("input_tokens", usage.get("prompt_tokens"))
                out_toks = usage.get("output_tokens", usage.get("completion_tokens"))
                reasoning_toks = usage.get("reasoning_tokens", 0)

                total_toks = usage.get("total_tokens")
                if total_toks is None and isinstance(in_toks, int) and isinstance(out_toks, int):
                    total_toks = in_toks + out_toks

                est_cost = _estimate_cost(in_toks, out_toks)

                # Extract
                terms = extract_hpo_terms(final_text)

                # If parser yields nothing, retry this run (do not count it)
                if not terms:
                    attempt += 1
                    continue

                # Success: log exactly once for this run_num
                cost_log_data.append([run_num, in_toks, out_toks, reasoning_toks, total_toks, est_cost])

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

        # If we still didn't get parseable output, do NOT advance run_num.
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
    output_summary = f"{base_name}_GPT_NO_TOOLS_summary.csv"
    with open(output_summary, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HPO ID", "Term Name", "Count (out of 50)", "Percentage"])
        for (hpo_id, term_name), count in term_counter.most_common():
            percentage = (count / NUM_RUNS) * 100
            writer.writerow([hpo_id, term_name, count, f"{percentage:.1f}%"])

    # === Export 2: Raw Run Log ===
    output_raw = f"{base_name}_GPT_NO_TOOLS_raw_log.csv"
    with open(output_raw, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "HPO ID", "Term Name", "Is Valid ID in DB?"])
        writer.writerows(raw_log_data)

    # === Export 3: Token/Cost Log (UPDATED HEADERS) ===
    output_cost = f"{base_name}_GPT_NO_TOOLS_cost_log.csv"
    with open(output_cost, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Run Number",
            "Input Tokens",
            "Output Tokens (Total)",
            "Reasoning Tokens (Subset of Output)",
            "Total Tokens",
            "Estimated Cost USD",
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
    # Filter out result files
    files = [f for f in txt_files if "summary" not in f and "raw_log" not in f and "hpo_terms" not in f and "cost_log" not in f]

    if not files:
        print("No input .txt files found.")
        return

    for f in files:
        process_file_reliability_no_tools(f)

if __name__ == "__main__":
    main()


