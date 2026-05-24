import os
import glob
import re
import csv
import json
import sys
import argparse
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional, Any
from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.tools import mcp  # Import the MCP tool helper

# =========================
# CONFIGURATION DEFAULTS
# =========================
XAI_API_KEY_DEFAULT = os.environ.get("XAI_API_KEY", "")
MCP_SERVER_URL_DEFAULT = "https://postsigmoidal-rosanne-interportal.ngrok-free.dev/sse"
HPO_JSON_FILE_DEFAULT = "hp.json"
NUM_RUNS_DEFAULT = 50
MAX_RETRIES_PER_RUN_DEFAULT = 8
EXTRACT_PASSES_DEFAULT = 1
MIN_PHENOTYPES_PER_RUN_DEFAULT = 5

# Cost tracking
PRICE_INPUT_PER_1M = 0.2
PRICE_OUTPUT_PER_1M = 0.5

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an expert Clinical Geneticist and phenotype extractor.

Goal: Produce a COMPREHENSIVE but ACCURATE list of patient phenotypic abnormalities from this note, mapped to HPO.
Do NOT invent findings. Only include a term if it is clearly supported in the note (explicitly stated or confidently inferable from objective data).

Critical inclusion/exclusion rules:
- Count ONLY findings in the PATIENT. Family history does NOT count unless the note explicitly states the patient has the finding.
- Exclude negated findings (e.g., "no seizures", "denies seizures", "negative for …").
- Exclude uncertainty ("possible", "concern for", "rule out") unless later confirmed.
- Do NOT add "expected" features of a suspected diagnosis unless documented in the patient.
- Avoid duplicative near-synonyms; prefer the most appropriate HPO concept(s). You may include BOTH a high-level parent and a specific child term if both are supported and non-redundant.

Inference rules (high recall, controlled):
You MAY infer a phenotype ONLY when objective data meet common clinical thresholds or patterns in the note:
- Microcephaly: OFC/HC < 3rd percentile or z ≤ -2
- Macrocephaly: OFC/HC > 97th percentile or z ≥ +2
- Short stature: height < 3rd percentile or z ≤ -2
- Tall stature: height > 97th percentile or z ≥ +2
- Underweight/failure to thrive: weight < ~3rd–5th percentile or z ≤ -2, or clear longitudinal concern
- Anemia: low hemoglobin/hematocrit; specify subtype only if stated or clearly supported (e.g., "normocytic")
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

# =========================
# EXTRACTION PROMPTS (Multi-pass support)
# =========================
EXTRACT_PROMPTS = [
    # Pass 1: Comprehensive extraction
    """Goal: Produce a COMPREHENSIVE but ACCURATE list of patient phenotypic abnormalities from this note, mapped to HPO.
Do NOT invent findings. Only include a term if it is clearly supported in the note (explicitly stated or confidently inferable from objective data).

Read the entire note and produce a comprehensive problem list of ONLY PATIENT phenotypic abnormalities.
Include both DIRECT (explicit) and INFERRED (threshold-based) findings per the system rules.
If you have access to tools, use `search_hpo_terms` to find the appropriate HPO terms.
Otherwise, use your internal knowledge to map to HPO.

Output format (ONLY):
- [HP:########] Official Term Name
""",
    # Pass 2: Catch missed items
    """Second pass: review the note again and find ADDITIONAL patient phenotypic abnormalities you may have missed.
Include subtle dysmorphology, growth, neurodevelopment, MSK, neuroimaging, heme, endocrine, cardiac, etc.
Do NOT repeat items you already listed if possible.

Output format (ONLY):
- [HP:########] Official Term Name
""",
    # Pass 3: Systematic sweep
    """Third pass: do a systematic organ-system sweep (growth, craniofacial, neurodevelopment, neuroimaging, MSK, cardiac, GI, renal/GU, endocrine, heme/immune, skin/hair).
Add any remaining patient phenotypic abnormalities (direct or inferred per thresholds).

Output format (ONLY):
- [HP:########] Official Term Name
""",
]

# =========================
# HPO VALIDATION
# =========================
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
            # Handle unexpected format: wrap dict in list, otherwise empty
            HPO_DATA = [data] if isinstance(data, dict) else []
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

# =========================
# USAGE/COST HELPERS
# =========================
def _get_usage_dict(resp):
    """Robust usage extraction including Reasoning Tokens."""
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
    """Calculates cost based on Total - Input = Billable Output."""
    if not (isinstance(in_toks, int) and isinstance(total_toks, int)):
        return None

    billable_output = total_toks - in_toks

    if not (isinstance(PRICE_INPUT_PER_1M, (int, float)) and isinstance(PRICE_OUTPUT_PER_1M, (int, float))):
        return None

    return (in_toks / 1_000_000.0) * PRICE_INPUT_PER_1M + (billable_output / 1_000_000.0) * PRICE_OUTPUT_PER_1M

# =========================
# PARSING HELPERS
# =========================
def extract_hpo_terms(text: str):
    """Extract HPO terms in format [HP:########] Term Name"""
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
    """Safe string conversion with truncation."""
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str, indent=2)
    except Exception:
        s = str(x)
    if limit and isinstance(s, str) and len(s) > limit:
        return s[:limit] + "\n...[TRUNCATED]..."
    return s

def normalize_text(s: str) -> str:
    """Normalize text for deduplication."""
    if not s:
        return ""
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = s.strip().lower()
    s = re.sub(r"[/,_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# EXTRACTION WITH MULTI-PASS SUPPORT
# =========================
def extract_phenotypes_for_run(
    note_content: str,
    client: Client,
    hpo_tool: Any,
    extract_passes: int,
    min_phenotypes_per_run: int,
    max_retries_per_run: int,
    model: str = "grok-4-1-fast-reasoning"
) -> Tuple[List[Tuple[int, str, str]], str]:
    """
    Extract phenotypes with multi-pass support.
    Returns: (list of (pass_num, hpo_id, term_name), raw_concatenated_text)
    """
    seen_norm: Set[str] = set()
    all_terms: List[Tuple[int, str, str]] = []
    raw_texts = []
    
    passes = min(extract_passes, len(EXTRACT_PROMPTS))
    
    for p in range(passes):
        user_prompt = EXTRACT_PROMPTS[p]
        attempt = 1
        last_error = None
        success = False
        
        while attempt <= max_retries_per_run:
            try:
                print(f"    Pass {p+1}/{passes}, attempt {attempt}/{max_retries_per_run}...", end="\r", flush=True)
                
                # Create fresh chat instance
                chat = client.chat.create(
                    model=model,
                    tools=[hpo_tool] if hpo_tool else None,
                )
                
                chat.append(system(SYSTEM_PROMPT))
                chat.append(user(note_content))
                chat.append(user(user_prompt))
                
                response = chat.sample()
                
                # Extract result
                result_text = getattr(response, "content", "")
                if not isinstance(result_text, str):
                    result_text = _safe_str(result_text)
                
                raw_texts.append(result_text)
                
                # Parse terms
                terms = extract_hpo_terms(result_text)
                
                # Add new terms (deduplicated)
                added = 0
                for hpo_id, term_name in terms:
                    norm = normalize_text(term_name)
                    if norm and norm not in seen_norm:
                        seen_norm.add(norm)
                        all_terms.append((p + 1, hpo_id, term_name))
                        added += 1
                
                print(f"    Pass {p+1}/{passes}: extracted {len(terms)} terms, added {added} new (total: {len(all_terms)})")
                success = True
                break
                
            except Exception as e:
                last_error = e
                attempt += 1
                continue
        
        if not success:
            print(f"    Pass {p+1}/{passes} failed after {max_retries_per_run} attempts: {last_error}")
            # Continue to next pass anyway
    
    raw_combined = "\n\n".join(raw_texts)
    return all_terms, raw_combined

# =========================
# MAIN PROCESSING
# =========================
def process_file_reliability(
    filepath: str,
    client: Client,
    hpo_tool: Any,
    num_runs: int,
    extract_passes: int,
    min_phenotypes_per_run: int,
    max_retries_per_run: int,
    mode: str,
    model: str = "grok-4-1-fast-reasoning"
):
    """Process a single file with reliability testing."""
    print(f"\n=== Processing: {filepath} ({num_runs} runs - mode={mode.upper()}) ===")

    with open(filepath, "r", encoding="utf-8") as f:
        note_content = f.read()

    term_counter = Counter()
    raw_log_data = []
    cost_log_data = []
    raw_text_log = []
    raw_debug_log = []

    run_num = 1
    while run_num <= num_runs:
        attempt = 1
        last_error = None
        success = False

        while attempt <= max_retries_per_run:
            try:
                print(f"  > Run {run_num}/{num_runs} (attempt {attempt}/{max_retries_per_run})...")
                
                # Extract with multi-pass support
                terms_with_pass, raw_text = extract_phenotypes_for_run(
                    note_content=note_content,
                    client=client,
                    hpo_tool=hpo_tool,
                    extract_passes=extract_passes,
                    min_phenotypes_per_run=min_phenotypes_per_run,
                    max_retries_per_run=max_retries_per_run,
                    model=model
                )
                
                # Check minimum phenotypes
                if len(terms_with_pass) < min_phenotypes_per_run:
                    print(f"    Too few terms: {len(terms_with_pass)} < {min_phenotypes_per_run}")
                    attempt += 1
                    continue
                
                # Log raw text and debug (use last response for cost tracking)
                raw_text_log.append([run_num, raw_text])
                raw_debug_log.append([run_num, f"Multi-pass extraction: {len(terms_with_pass)} total terms"])
                
                # Note: Cost tracking would need to aggregate across all passes
                # For now, we'll just log a placeholder
                cost_log_data.append([run_num, 0, 0, 0, 0, 0.0])  # Placeholder
                
                # Update counter
                for pass_num, hpo_id, term_name in terms_with_pass:
                    term_counter.update([(hpo_id, term_name)])
                    is_valid = str(hpo_id in VALID_HPO_IDS)
                    raw_log_data.append([run_num, pass_num, hpo_id, term_name, is_valid])
                
                success = True
                break

            except Exception as e:
                last_error = e
                attempt += 1
                continue

        if not success:
            if last_error is not None:
                print(f"\n  Run {run_num} failed after {max_retries_per_run} attempts (last error: {last_error}). Retrying run {run_num}...")
            else:
                print(f"\n  Run {run_num} produced no parseable terms after {max_retries_per_run} attempts. Retrying run {run_num}...")
            continue

        run_num += 1

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # Export CSVs
    output_summary = f"{base_name}_GROK_{mode.upper()}_summary.csv"
    with open(output_summary, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HPO ID", "Term Name", "Count (out of runs)", "Percentage"])
        for (hpo_id, term_name), count in term_counter.most_common():
            percentage = (count / num_runs) * 100
            writer.writerow([hpo_id, term_name, count, f"{percentage:.1f}%"])

    output_raw = f"{base_name}_GROK_{mode.upper()}_raw_log.csv"
    with open(output_raw, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "Pass", "HPO ID", "Term Name", "Is Valid ID in DB?"])
        writer.writerows(raw_log_data)

    output_cost = f"{base_name}_GROK_{mode.upper()}_cost_log.csv"
    with open(output_cost, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Run Number",
            "Input Tokens",
            "Output Tokens",
            "Reasoning Tokens",
            "Total Tokens",
            "Estimated Cost USD"
        ])
        writer.writerows(cost_log_data)

    output_raw_text = f"{base_name}_GROK_{mode.upper()}_raw_text.csv"
    with open(output_raw_text, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "Raw Output"])
        writer.writerows(raw_text_log)

    output_debug = f"{base_name}_GROK_{mode.upper()}_debug.csv"
    with open(output_debug, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run Number", "Debug"])
        writer.writerows(raw_debug_log)

    print(f"\n  Saved summary to {output_summary}")
    print(f"  Saved raw log to {output_raw}")
    print(f"  Saved cost log to {output_cost}")
    print(f"  Saved raw text to {output_raw_text}")
    print(f"  Saved debug log to {output_debug}")

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Grok HPO extraction with optional MCP tools and multi-pass support")
    
    # Core settings
    parser.add_argument("--api-key", type=str, default=XAI_API_KEY_DEFAULT, help="xAI API key")
    parser.add_argument("--model", type=str, default="grok-4-1-fast-reasoning", help="Grok model to use")
    parser.add_argument("--runs", type=int, default=NUM_RUNS_DEFAULT, help="Number of runs per file")
    parser.add_argument("--hpo", type=str, default=HPO_JSON_FILE_DEFAULT, help="HPO JSON file path")
    
    # Mode settings
    parser.add_argument(
        "--mode",
        choices=["with_mcp", "no_mcp"],
        default="with_mcp",
        help="Run with or without MCP tools"
    )
    parser.add_argument("--mcp-url", type=str, default=MCP_SERVER_URL_DEFAULT, help="MCP server URL (for with_mcp mode)")
    
    # Extraction settings
    parser.add_argument(
        "--extract-passes",
        type=int,
        default=EXTRACT_PASSES_DEFAULT,
        help="Number of extraction passes (1-3)"
    )
    parser.add_argument(
        "--min-phenotypes",
        type=int,
        default=MIN_PHENOTYPES_PER_RUN_DEFAULT,
        help="Minimum phenotypes required per run"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_PER_RUN_DEFAULT,
        help="Maximum retries per run"
    )
    
    args = parser.parse_args()
    
    # Load HPO data
    load_data(args.hpo)
    if not HPO_DATA:
        return
    
    # Initialize client
    try:
        client = Client(api_key=args.api_key)
        print(f"Initialized xAI client with model: {args.model}")
    except Exception as e:
        print(f"Error initializing client: {e}")
        return
    
    # Setup MCP tool if needed
    hpo_tool = None
    if args.mode == "with_mcp":
        try:
            hpo_tool = mcp(server_url=args.mcp_url)
            print(f"MCP tool configured: {args.mcp_url}")
        except Exception as e:
            print(f"Warning: Could not configure MCP tool: {e}")
            print("Continuing without MCP tools...")
            args.mode = "no_mcp"
    
    # Find files to process
    txt_files = glob.glob("*.txt")
    files_to_process = [
        f for f in txt_files
        if "summary" not in f and "raw_log" not in f and "hpo_terms" not in f and "cost_log" not in f
    ]

    if not files_to_process:
        print("No clinical note .txt files found.")
        return

    print(f"\n=== Configuration ===")
    print(f"Mode: {args.mode.upper()}")
    print(f"Model: {args.model}")
    print(f"Runs per file: {args.runs}")
    print(f"Extract passes: {args.extract_passes}")
    print(f"Min phenotypes per run: {args.min_phenotypes}")
    print(f"Max retries: {args.max_retries}")
    if args.mode == "with_mcp":
        print(f"MCP URL: {args.mcp_url}")
    print(f"Files to process: {len(files_to_process)}")
    print("=" * 50)

    for filepath in files_to_process:
        process_file_reliability(
            filepath=filepath,
            client=client,
            hpo_tool=hpo_tool,
            num_runs=args.runs,
            extract_passes=args.extract_passes,
            min_phenotypes_per_run=args.min_phenotypes,
            max_retries_per_run=args.max_retries,
            mode=args.mode,
            model=args.model
        )

if __name__ == "__main__":
    main()
