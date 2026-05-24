import os
import glob
import os
import glob
import re
import csv
import json
import sys
from collections import Counter
from typing import List, Dict, Set
import time  # <--- Added for rate limiting
import anthropic  # Official Anthropic client

# === CONFIGURATION ===
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")     # PUT YOUR KEY
HPO_JSON_FILE = "hp.json"
NUM_RUNS = 50
MODEL_NAME = "claude-sonnet-4-5-20250929"

# === COST TRACKING ===
# Using Sonnet 4.5 pricing ($3/$15) as per your table
PRICE_INPUT_PER_1M = 3
PRICE_OUTPUT_PER_1M = 15

# =====================

# === VALIDATION LOGIC (same as Grok)
HPO_DATA: List[Dict[str, str]] = []
VALID_HPO_IDS: Set[str] = set()

def load_data(filepath: str = "hp.json"):
    """Loads HPO data to validate results for the log."""
    global HPO_DATA, VALID_HPO_IDS
    try:
        print(f"Loading HPO data from {filepath} for validation...")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "graphs" in data:
            HPO_DATA = data["graphs"][0]["nodes"]
        elif isinstance(data, list):
            HPO_DATA = data
        else:
            HPO_DATA = data

        for term in HPO_DATA:
            raw_id = term.get("id", "")
            if "HP_" in raw_id:
                clean_id = "HP:" + raw_id.split("HP_")[-1]
                VALID_HPO_IDS.add(clean_id)
            else:
                VALID_HPO_IDS.add(raw_id)

        print(f"SUCCESS: Loaded {len(HPO_DATA)} terms. Validation ready.")

    except Exception as e:
        print(f"ERROR loading JSON: {e}")
        sys.exit(1)



# === Claude Client ===
try:
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
except Exception as e:
    print(f"Error initializing Anthropic client. Is your CLAUDE_API_KEY correct? {e}")
    sys.exit(1)


# === SYSTEM PROMPT (mirrors Grok-NO-TOOLS exactly)
SYSTEM_PROMPT_NO_TOOLS = """
You are an expert Clinical Geneticist and phenotype extractor. Your task is to analyze the following clinical
note and identify the patient’s phenotypic abnormalities, mapped to HPO.

Goal: MAXIMIZE RECALL while MAINTAINING HIGH PRECISION.
Only include a term if it is clearly supported in the note (explicitly stated OR confidently inferable from objective data using standard clinical interpretations).
Do NOT invent findings.

Global rules (apply to everything):
- PATIENT ONLY: Family history does NOT count unless the note explicitly states the patient has the finding.
- NEGATIONS do NOT count (e.g., “no seizures”, “denies seizures”, “negative for…”).
- UNCERTAINTY does NOT count (“possible”, “concern for”, “rule out”) unless later confirmed in the note.
- Do NOT add typical features of a suspected syndrome unless documented for the patient.
- Prefer a single best HPO term per concept; include a more specific subtype only if clearly supported.

Inference rules (allowed, but controlled):
You MAY infer a phenotype ONLY when objective data meet common clinical thresholds/patterns in the note:
- Microcephaly: OFC/HC < 3rd percentile or z ≤ -2
- Macrocephaly: OFC/HC > 97th percentile or z ≥ +2
- Short stature: height < 3rd percentile or z ≤ -2
- Tall stature: height > 97th percentile or z ≥ +2
- Underweight/failure to thrive: weight < ~3rd–5th percentile or z ≤ -2, or clear longitudinal concern
- Anemia: low hemoglobin/hematocrit; specify subtype only if stated or clearly supported
- Hypothyroidism: elevated TSH with low free T4 (or explicit diagnosis)

Workflow (MUST follow):
1) Read the ENTIRE note.
2) Internally create a problem list of abnormalities (growth, development, neuro, MSK, cardiac, endocrine, heme, dysmorphology).
3) For each abnormality, translate it into an interpreted phenotype concept.
4) Assign the official HPO ID for each included concept using internal knowledge.
   - If you are not confident of the exact HPO ID, OMIT the term rather than guessing.
5) Non-redundancy:
   - Output ONE best HPO term per concept.
   - Prefer higher-level terms unless a specific subtype is explicitly supported.
6) Scope:
   - Focus on persistent/high-impact phenotypes.
   - Limit to at most 25 terms. If you have more, keep the most important and best-supported.

Output formatting (critical):
- You may optionally provide ONE short "Clinical Synthesis" paragraph.
- After that, output exactly ONE final list of HPO terms and IDs.
- Do NOT output intermediate lists or commentary.

Final output format (must match exactly):
<FINAL_HPO_LIST>
- [HP:0001234] Term Name
- [HP:0005678] Another Term Name
</FINAL_HPO_LIST>
"""




# === Regex extractor (same as Grok, OpenAI, Claude-with-tools)
def extract_hpo_terms(text: str) -> List[tuple]:
    """
    Extract ONLY the final contiguous block of HPO bullet lines, e.g.:

    - [HP:0001234] Term
    - [HP:0005678] Another term

    This avoids earlier "draft" lists that Claude may later revise.
    Also deduplicates HPO IDs within a single run.
    """
    lines = text.splitlines()
    blocks: List[List[str]] = []
    current_block: List[str] = []

    hpo_line_re = re.compile(r'\s*-\s*\[(HP:\d+)\]\s*(.+)')

    for line in lines:
        m = hpo_line_re.match(line)
        if m:
            current_block.append(line)
        else:
            if current_block:
                blocks.append(current_block)
                current_block = []

    # Catch a trailing block at EOF
    if current_block:
        blocks.append(current_block)

    if not blocks:
        return []

    # Take ONLY the last block (the final, cleaned-up list)
    last_block = blocks[-1]

    terms = []
    for line in last_block:
        m = hpo_line_re.match(line)
        if not m:
            continue
        hpo_id, name = m.groups()
        terms.append((hpo_id, name.strip()))

    # Deduplicate by HPO ID within a single run, preserve order
    seen_ids = set()
    deduped_terms = []
    for hpo_id, name in terms:
        if hpo_id not in seen_ids:
            seen_ids.add(hpo_id)
            deduped_terms.append((hpo_id, name))

    return deduped_terms


def process_file_no_tools(filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]  # <-- FIX: define early

    print(f"\n=== Processing: {filepath} ({NUM_RUNS} runs - CLAUDE THINKING NO TOOLS) ===")

    with open(filepath, "r", encoding="utf-8") as f:
        note_content = f.read()

    term_counter = Counter()
    raw_log_data = []
    raw_text_log = []   # collect raw text per run
    cost_log = []       # per-run token/cost tracking

    for i in range(1, NUM_RUNS + 1):
        try:
            print(f"  > Run {i}/{NUM_RUNS}...", end="\r")

            with client.messages.stream(
                model=MODEL_NAME,
                system=SYSTEM_PROMPT_NO_TOOLS,
                messages=[{"role": "user", "content": note_content}],
                thinking={"type": "enabled", "budget_tokens": 16000},
                max_tokens=40000,
                stop_sequences=["</FINAL_HPO_LIST>"],
            ) as stream:
                response = stream.get_final_message()


            # === TOKEN + COST CAPTURE (UPDATED) ===
            usage = getattr(response, "usage", None)
            
            in_toks = getattr(usage, "input_tokens", 0)
            out_toks = getattr(usage, "output_tokens", 0)
            
            # Extract Thinking Tokens for Scientific Analysis
            # Note: These are INCLUDED in output_tokens for billing, but useful to separate for analysis
            think_toks = getattr(usage, "thinking_tokens", 0)

            total_toks = in_toks + out_toks

            est_cost = None
            if (
                isinstance(PRICE_INPUT_PER_1M, (int, float))
                and isinstance(PRICE_OUTPUT_PER_1M, (int, float))
            ):
                # Thinking tokens are billed as output tokens at $15/1M
                est_cost = (in_toks / 1_000_000.0) * PRICE_INPUT_PER_1M + (out_toks / 1_000_000.0) * PRICE_OUTPUT_PER_1M

            # Log including thinking tokens
            cost_log.append([i, in_toks, out_toks, think_toks, total_toks, est_cost])

            # Gather only text blocks (ignore thinking blocks)
            final_text = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    final_text += block.text

            raw_text_log.append([i, final_text])

            # Extract HPO terms
            terms = extract_hpo_terms(final_text)
            term_counter.update(terms)

            # Log structured HPO data
            if not terms:
                raw_log_data.append([i, "NO_TERMS_FOUND", "N/A", "False"])
            else:
                for hpo_id, term_name in terms:
                    is_valid = str(hpo_id in VALID_HPO_IDS)
                    raw_log_data.append([i, hpo_id, term_name, is_valid])

            # Sleep slightly longer for thinking models
            time.sleep(5)

        except Exception as e:
            print(f"\n  Error on run {i}: {e}")
            raw_log_data.append([i, "ERROR", str(e), "False"])
            raw_text_log.append([i, f"ERROR: {e}"])
            cost_log.append([i, None, None, None, None, None])
            time.sleep(60)

    # Summary
    output_summary = f"{base}_CLAUDE_NO_TOOLS_summary.csv"
    with open(output_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["HPO ID", "Term Name", "Count (out of 50)", "Percentage"])
        for (hpo_id, name), count in term_counter.most_common():
            pct = (count / NUM_RUNS) * 100
            w.writerow([hpo_id, name, count, f"{pct:.1f}%"])

    # Raw structured HPO log
    output_raw = f"{base}_CLAUDE_NO_TOOLS_raw_log.csv"
    with open(output_raw, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Run Number", "HPO ID", "Term Name", "Is Valid ID in DB?"])
        w.writerows(raw_log_data)

    # Raw text log
    output_raw_text = f"{base}_CLAUDE_NO_TOOLS_raw_text.csv"
    with open(output_raw_text, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Run Number", "Raw Output"])
        w.writerows(raw_text_log)

    # Cost/token log (Updated headers)
    output_cost_log = f"{base}_CLAUDE_NO_TOOLS_cost_log.csv"
    with open(output_cost_log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Run Number", 
            "Input Tokens", 
            "Output Tokens (Includes Thinking)", 
            "Thinking Tokens (Subset)", # <--- Valuable metric for paper
            "Total Tokens", 
            "Estimated Cost USD"
        ])
        w.writerows(cost_log)

    print(f"\n  Saved summary to {output_summary}")
    print(f"  Saved raw log to {output_raw}")
    print(f"  Saved raw text to {output_raw_text}")
    print(f"  Saved cost log to {output_cost_log}")


def main():
    load_data(HPO_JSON_FILE)
    if not HPO_DATA:
        return

    txt_files = glob.glob("*.txt")
    files = [
        f for f in txt_files
        if "summary" not in f and "raw_log" not in f and "hpo_terms" not in f
    ]

    if not files:
        print("No clinical note .txt files found.")
        return

    for f in files:
        process_file_no_tools(f)


if __name__ == "__main__":
    main()