


import os
import glob
import re
import csv
import json
import sys
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional
import anthropic  # Official Anthropic client
import time  # rate limiting

# === CONFIGURATION ===
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MCP_SERVER_URL = "https://postsigmoidal-rosanne-interportal.ngrok-free.dev/sse"
HPO_JSON_FILE = "hp.json"
NUM_RUNS = 60
MODEL_NAME = "claude-sonnet-4-5-20250929"  # Using your specified ID

# Output control
MAX_TOKENS = 40000
THINKING_BUDGET = 16000

# === COST TRACKING (Updated based on your provided table) ===
PRICE_INPUT_PER_1M = 3.0    # $3.00 per 1M input
PRICE_OUTPUT_PER_1M = 15.0  # $15.00 per 1M output (includes thinking tokens)
# =====================

# === VALIDATION LOGIC (same as Grok) ===
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


# === Anthropic Client ===
if not CLAUDE_API_KEY:
    print("Error: CLAUDE_API_KEY is empty. Set it as an environment variable (CLAUDE_API_KEY).")
    sys.exit(1)

try:
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
except Exception as e:
    print(f"Error initializing Anthropic client. Is your CLAUDE_API_KEY correct? {e}")
    sys.exit(1)


# === SYSTEM PROMPT (mirrors Grok version) ===
SYSTEM_PROMPT = """
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
4) Use the `search_hpo_terms` tool ONLY for interpreted phenotype concepts (not raw numbers, treatments, devices).
   - BAD: `search_hpo_terms("TSH 14")`
   - BAD: `search_hpo_terms("CPAP")`
   - GOOD: `search_hpo_terms("Hypothyroidism")`
   - GOOD: `search_hpo_terms("Microcephaly")`
5) Non-redundancy:
   - Output ONE best HPO term per concept.
   - Prefer higher-level terms unless a specific subtype is explicitly supported.
6) Scope:
   - Focus on persistent/high-impact phenotypes.
   - Limit to at most 25 terms. If you have more, keep the most important and best-supported.

Output formatting (critical):
- Output ONLY one final list of HPO terms and IDs.
- Do NOT output intermediate lists or commentary.

Final output format (must match exactly):
<FINAL_HPO_LIST>
- [HP:0001234] Term Name
- [HP:0005678] Another Term Name
</FINAL_HPO_LIST>
"""


def extract_usage_fields(resp) -> Dict[str, int]:
    """Robustly extract usage fields including thinking tokens (if available)."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}

    out = {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "thinking_tokens": getattr(usage, "thinking_tokens", 0) if hasattr(usage, "thinking_tokens") else 0,
    }
    return out


def estimate_cost_usd(usage: Dict[str, int]) -> Optional[float]:
    """Simple estimator using input/output token prices."""
    in_toks = usage.get("input_tokens", 0)
    out_toks = usage.get("output_tokens", 0)

    if not isinstance(PRICE_INPUT_PER_1M, (int, float)) or not isinstance(PRICE_OUTPUT_PER_1M, (int, float)):
        return None

    return (in_toks / 1_000_000.0) * PRICE_INPUT_PER_1M + (out_toks / 1_000_000.0) * PRICE_OUTPUT_PER_1M


def extract_hpo_terms(text: str) -> List[Tuple[str, str]]:
    """
    Extract HPO lines in either of these formats:
      - [HP:0001234] Term Name
        [HP:0001234] Term Name
    Returns list of (ID, Name), deduped by ID (order preserved).
    """
    if not text:
        return []

    line_re = re.compile(r"^\s*(?:-\s*)?\[(HP:\d+)\]\s*(.+?)\s*$", re.MULTILINE)
    matches = line_re.findall(text)

    seen = set()
    out: List[Tuple[str, str]] = []
    for hpo_id, name in matches:
        if hpo_id not in seen:
            seen.add(hpo_id)
            out.append((hpo_id, name.strip()))
    return out


def run_claude_with_stream(note_content: str, mcp_server_config: List[Dict[str, str]]):
    """
    Streaming is required for requests that may take >10 minutes.
    Returns (final_text, final_message_object).
    """
    final_text = ""

    with client.messages.stream(
        model=MODEL_NAME,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": note_content}],
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
        extra_headers={"anthropic-beta": "mcp-client-2025-04-04"},
        extra_body={"mcp_servers": mcp_server_config},
    ) as stream:
        # text_stream yields only "text" deltas (not tool-use blocks, not thinking)
        for chunk in stream.text_stream:
            final_text += chunk

        # Get the final assembled message (for usage, etc.)
        if hasattr(stream, "get_final_message"):
            final_message = stream.get_final_message()
        else:
            final_message = getattr(stream, "final_message", None)

    return final_text, final_message


def process_file_reliability(filepath: str):
    """
    For a single .txt note:
    - Run Claude (Thinking Mode) + MCP NUM_RUNS times (STREAMING)
    - Extract HPO IDs from free text
    - Aggregate counts and log raw runs
    """
    print(f"\n=== Processing: {filepath} ({NUM_RUNS} runs - CLAUDE THINKING + MCP) ===")

    with open(filepath, "r", encoding="utf-8") as f:
        note_content = f.read()

    term_counter = Counter()
    raw_log_data = []
    cost_log = []

    mcp_server_config = [
        {
            "type": "url",
            "name": "hpo_server",
            "url": MCP_SERVER_URL,
        }
    ]

    for i in range(1, NUM_RUNS + 1):
        try:
            print(f"  > Run {i}/{NUM_RUNS}...", end="\r")

            final_text, final_msg = run_claude_with_stream(note_content, mcp_server_config)

            usage = extract_usage_fields(final_msg) if final_msg is not None else {}
            in_toks = usage.get("input_tokens", None)
            out_toks = usage.get("output_tokens", None)
            think_toks = usage.get("thinking_tokens", None)
            cache_create = usage.get("cache_creation_input_tokens", None)
            cache_read = usage.get("cache_read_input_tokens", None)

            total_toks = None
            if isinstance(in_toks, int) and isinstance(out_toks, int):
                total_toks = in_toks + out_toks

            est_cost = estimate_cost_usd(usage) if usage else None
            cost_log.append([i, in_toks, out_toks, think_toks, cache_create, cache_read, total_toks, est_cost])

            if not final_text.strip():
                msg = "NO_TEXT_RESPONSE"
                raw_log_data.append([i, msg, "N/A", "False"])
                time.sleep(5)
                continue

            terms = extract_hpo_terms(final_text)
            term_counter.update(terms)

            if not terms:
                raw_log_data.append([i, "NO_TERMS_FOUND", "N/A", "False"])
            else:
                for hpo_id, term_name in terms:
                    is_valid = str(hpo_id in VALID_HPO_IDS)
                    raw_log_data.append([i, hpo_id, term_name, is_valid])

            time.sleep(5)

        except Exception as e:
            print(f"\n  Error on run {i}: {e}")
            raw_log_data.append([i, "ERROR", str(e), "False"])
            cost_log.append([i, None, None, None, None, None, None, None])
            time.sleep(60)

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    output_summary = f"{base_name}_CLAUDE_THINKING_WITH_MCP_summary.csv"
    with open(output_summary, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HPO ID", "Term Name", "Count (out of 60)", "Percentage"])
        for (hpo_id, term_name), count in term_counter.most_common():
            percentage = (count / NUM_RUNS) * 100
            writer.writerow([hpo_id, term_name, count, f"{percentage:.1f}%"])

    output_raw = f"{base_name}_CLAUDE_THINKING_WITH_MCP_raw_log.csv"
    with open(output_raw, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "HPO ID", "Term Name", "Is Valid ID in DB?"])
        writer.writerows(raw_log_data)

    output_cost = f"{base_name}_CLAUDE_THINKING_WITH_MCP_cost_log.csv"
    with open(output_cost, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Run Number",
                "Input Tokens",
                "Output Tokens",
                "Thinking Tokens (Included in Output)",
                "Cache Creation",
                "Cache Read",
                "Total Tokens (In+Out)",
                "Estimated Cost USD",
            ]
        )
        writer.writerows(cost_log)

    print(f"\n  Saved summary to {output_summary}")
    print(f"  Saved raw log to {output_raw}")
    print(f"  Saved cost log to {output_cost}")


def main():
    load_data(HPO_JSON_FILE)
    if not HPO_DATA:
        return

    txt_files = glob.glob("*.txt")
    files_to_process = [
        f
        for f in txt_files
        if "summary" not in f
        and "raw_log" not in f
        and "hpo_terms" not in f
        and "cost_log" not in f
    ]

    if not files_to_process:
        print("No clinical note .txt files found.")
        return

    for filepath in files_to_process:
        process_file_reliability(filepath)


if __name__ == "__main__":
    main()

