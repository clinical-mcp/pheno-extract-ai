import os
import glob
import re
import csv
import json
import sys
from collections import Counter
from typing import List, Dict, Set
from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.tools import mcp  # Import the MCP tool helper

# --- CONFIGURATION (MUST EDIT) ---

# === CONFIGURATION ===
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
# Paste your ngrok URL from the other terminal
MCP_SERVER_URL = "https://postsigmoidal-rosanne-interportal.ngrok-free.dev/sse"
HPO_JSON_FILE = "hp.json"
NUM_RUNS = 50
# =====================

# === COST TRACKING ===
PRICE_INPUT_PER_1M = 0.2   
PRICE_OUTPUT_PER_1M = 0.5  
# =====================

# === VALIDATION LOGIC ===
HPO_DATA: List[Dict[str, str]] = []
VALID_HPO_IDS: Set[str] = set()

def load_data(filepath: str = "hp.json"):
    """Loads HPO data to validate results for the log."""
    global HPO_DATA, VALID_HPO_IDS
    try:
        print(f"Loading HPO data from {filepath} for validation...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'graphs' in data:
            HPO_DATA = data['graphs'][0]['nodes']
        elif isinstance(data, list):
            HPO_DATA = data
        else:
            HPO_DATA = data

        # Create valid ID set
        for term in HPO_DATA:
            raw_id = term.get('id', '')
            if 'HP_' in raw_id:
                clean_id = 'HP:' + raw_id.split('HP_')[-1]
                VALID_HPO_IDS.add(clean_id)
            else:
                VALID_HPO_IDS.add(raw_id)

        print(f"SUCCESS: Loaded {len(HPO_DATA)} terms. Validation ready.")
    except Exception as e:
        print(f"ERROR loading JSON: {e}")
        sys.exit(1)

# Initialize the official xAI Client
try:
    client = Client(api_key=XAI_API_KEY)
except Exception as e:
    print(f"Error initializing client. Is your XAI_API_KEY correct? {e}")
    exit()

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

def _get_usage_dict(resp):
    """
    Robust usage extraction including Reasoning Tokens.
    """
    if resp is None:
        return {}
    u = getattr(resp, "usage", None)
    if u is None:
        u = getattr(resp, "token_usage", None)
    if u is None and hasattr(resp, "metadata"):
        u = getattr(resp.metadata, "usage", None)
    if u is None:
        return {}

    if isinstance(u, dict):
        return u

    out = {}
    for k in ["input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"]:
        v = getattr(u, k, None)
        if isinstance(v, int):
            out[k] = v
            
    # === NEW: Capture Reasoning Tokens ===
    out["reasoning_tokens"] = getattr(u, "reasoning_tokens", 0)

    # Normalize aliases
    if "input_tokens" not in out and "prompt_tokens" in out:
        out["input_tokens"] = out["prompt_tokens"]
    if "output_tokens" not in out and "completion_tokens" in out:
        out["output_tokens"] = out["completion_tokens"]
        
    # Recalculate total if missing
    if "total_tokens" not in out and isinstance(out.get("input_tokens"), int) and isinstance(out.get("output_tokens"), int):
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"] + out.get("reasoning_tokens", 0)
        
    return out

def _estimate_cost(in_toks, total_toks):
    """
    Calculates cost based on Total - Input = Billable Output.
    This handles cases where reasoning tokens are billed as output but reported separately.
    """
    if not (isinstance(in_toks, int) and isinstance(total_toks, int)):
        return None
        
    billable_output = total_toks - in_toks
    
    if not (isinstance(PRICE_INPUT_PER_1M, (int, float)) and isinstance(PRICE_OUTPUT_PER_1M, (int, float))):
        return None
        
    return (in_toks / 1_000_000.0) * PRICE_INPUT_PER_1M + (billable_output / 1_000_000.0) * PRICE_OUTPUT_PER_1M

def extract_hpo_terms(text: str):
    if not text:
        return []
    line_re = re.compile(r'^\s*(?:-\s*)?\[(HP:\d+)\]\s*(.+?)\s*$', re.MULTILINE)
    matches = line_re.findall(text)
    seen = set()
    out = []
    for hpo_id, name in matches:
        if hpo_id not in seen:
            seen.add(hpo_id)
            out.append((hpo_id, name.strip()))
    return out

def _safe_str(x, limit=200000):
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str, indent=2)
    except Exception:
        s = str(x)
    if limit and isinstance(s, str) and len(s) > limit:
        return s[:limit] + "\n...[TRUNCATED]..."
    return s

def process_file_reliability(filepath):
    print(f"\n=== Processing: {filepath} ({NUM_RUNS} runs - WITH NO TOOLS) ===")

    with open(filepath, "r", encoding="utf-8") as f:
        note_content = f.read()

    term_counter = Counter()

    raw_log_data = []
    cost_log_data = []
    raw_text_log = []
    raw_debug_log = []

    # 1. Define tool

    for i in range(1, NUM_RUNS + 1):
        try:
            print(f"  > Run {i}/{NUM_RUNS}...", end="\r")

            # 2. Create fresh chat instance
            chat = client.chat.create(
                model="grok-4-1-fast-reasoning",
                # REMOVED: temperature=0.2 (Use default for reasoning)
            )

            chat.append(system(SYSTEM_PROMPT_NO_TOOLS))
            chat.append(user(note_content))

            response = chat.sample()

            # ---- RAW CAPTURE ----
            result_text = getattr(response, "content", "")
            if not isinstance(result_text, str):
                result_text = _safe_str(result_text)

            raw_text_log.append([i, result_text])
            raw_debug_log.append([i, _safe_str(getattr(response, "__dict__", response))])

            # ---- USAGE/COST ----
            u = _get_usage_dict(response)
            in_toks = u.get("input_tokens", 0)
            out_toks = u.get("output_tokens", 0)
            reasoning_toks = u.get("reasoning_tokens", 0) # <--- Critical for Paper
            tot_toks = u.get("total_tokens", 0)
            
            # Robust cost calculation
            est_cost = _estimate_cost(in_toks, tot_toks)
            
            cost_log_data.append([i, in_toks, out_toks, reasoning_toks, tot_toks, est_cost])

            # ---- EXTRACT ----
            terms = extract_hpo_terms(result_text)
            term_counter.update(terms)

            if not terms:
                raw_log_data.append([i, "NO_TERMS_FOUND", "N/A", "False"])
            else:
                for hpo_id, term_name in terms:
                    is_valid = str(hpo_id in VALID_HPO_IDS)
                    raw_log_data.append([i, hpo_id, term_name, is_valid])

        except Exception as e:
            print(f"\n  Error on run {i}: {e}")
            raw_log_data.append([i, "ERROR", str(e), "False"])
            cost_log_data.append([i, None, None, None, None, None])
            raw_text_log.append([i, f"ERROR: {e}"])
            raw_debug_log.append([i, f"ERROR: {e}"])

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # === Export 1: Aggregate Summary ===
    output_summary = f"{base_name}_GROK_NO_TOOLS_summary.csv"
    with open(output_summary, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HPO ID", "Term Name", "Count (out of 50)", "Percentage"])
        for (hpo_id, term_name), count in term_counter.most_common():
            percentage = (count / NUM_RUNS) * 100
            writer.writerow([hpo_id, term_name, count, f"{percentage:.1f}%"])

    # === Export 2: Raw Run Log ===
    output_raw = f"{base_name}_GROK_NO_TOOLS_raw_log.csv"
    with open(output_raw, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "HPO ID", "Term Name", "Is Valid ID in DB?"])
        writer.writerows(raw_log_data)

    # === Export 3: Cost Log ===
    output_cost = f"{base_name}_GROK_NO_TOOLS_cost_log.csv"
    with open(output_cost, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Run Number",
            "Input Tokens",
            "Output Tokens",
            "Reasoning Tokens", # <--- New column
            "Total Tokens",
            "Estimated Cost USD"
        ])
        writer.writerows(cost_log_data)

    # === Export 4 & 5: Debug ===
    output_raw_text = f"{base_name}_GROK_NO_TOOLS_raw_text.csv"
    with open(output_raw_text, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "Raw Output"])
        writer.writerows(raw_text_log)

    output_debug = f"{base_name}_GROK_NO_TOOLS_debug.csv"
    with open(output_debug, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "Debug"])
        writer.writerows(raw_debug_log)

    print(f"\n  Saved summary to {output_summary}")
    print(f"  Saved raw log to {output_raw}")
    print(f"  Saved cost log to {output_cost}")
    print(f"  Saved raw text to {output_raw_text}")
    print(f"  Saved debug log to {output_debug}")


def main():
    load_data(HPO_JSON_FILE)
    if not HPO_DATA:
        return

    txt_files = glob.glob("*.txt")
    files_to_process = [
        f for f in txt_files
        if "summary" not in f and "raw_log" not in f and "hpo_terms" not in f and "cost_log" not in f
    ]

    if not files_to_process:
        print("No clinical note .txt files found.")
        return

    for filepath in files_to_process:
        process_file_reliability(filepath)

if __name__ == "__main__":
    main()