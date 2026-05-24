import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from xai_sdk import Client
from xai_sdk.chat import system, user


def setup_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("categorize_terms")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG if verbose else logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def note_sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def sniff_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(8192)
    if "\t" in sample and sample.count("\t") > sample.count(","):
        return "\t"
    return ","


def read_table(path: str, delimiter: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Could not read header row from file.")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def write_table(path: str, delimiter: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_term(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_cache(cache_path: str) -> Dict[str, Any]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache(cache_path: str, cache: Dict[str, Any]) -> None:
    if not cache_path:
        return
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, cache_path)


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not parse JSON object from Grok response.")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    return False


def build_system_prompt() -> str:
    return (
        "You are an expert pediatric genetics chart reviewer.\n"
        "You will answer a single yes/no question about one phenotype term.\n"
        "Answer ONLY with valid JSON: {\"answer\": true/false}.\n\n"
        "Rules:\n"
        "- If uncertain, answer false.\n"
        "- Mentions in family history do NOT count for the patient unless explicitly stated.\n"
        "- Negated findings (e.g., 'no seizures') do NOT count.\n"
        "- If asked about a specific section, only answer true if the finding appears in that section.\n"
        "- For section questions, count DIRECT statements OR clear INFERRED evidence found in that section.\n"
        "- If asked about inferred findings, answer true ONLY if inferred from objective data and NOT explicitly stated.\n"
        "- If asked about active problems, answer true ONLY if current/active, not historical/resolved.\n"
        "Return JSON only."
    )


def build_user_prompt(term: str, official_terms: str, hpo_id: str, question: str, note_text: str) -> str:
    return (
        f"PHENOTYPE TERM (canonical): {term}\n"
        f"OFFICIAL/ORIGINAL TERM(S): {official_terms}\n"
        f"HPO ID: {hpo_id}\n\n"
        f"QUESTION: {question}\n\n"
        "Interpretation rules:\n"
        "- If uncertain, answer false.\n"
        "- Mentions in family history do NOT count for the patient unless explicitly stated.\n"
        "- Negated findings (e.g., 'no seizures') do NOT count.\n"
        "- For section questions, answer true if the phenotype is directly stated OR clearly inferred from data within that section.\n"
        "- If asked about inferred findings, answer true ONLY if inferred from objective data and NOT explicitly stated.\n"
        "- If asked about active problems, answer true ONLY if current/active, not historical/resolved.\n\n"
        "CLINICAL NOTE:\n"
        "-----\n"
        f"{note_text}\n"
        "-----\n\n"
        "Return JSON exactly as {\"answer\": true/false}."
    )


def grok_yes_no(
    logger: logging.Logger,
    client: Client,
    model: str,
    system_prompt: str,
    term: str,
    official_terms: str,
    hpo_id: str,
    question: str,
    note_text: str,
    max_retries: int = 4,
    retry_sleep_s: float = 2.0,
) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            chat = client.chat.create(model=model)
            chat.append(system(system_prompt))
            chat.append(user(build_user_prompt(term, official_terms, hpo_id, question, note_text)))
            response = chat.sample()
            data = extract_first_json_object(response.content)
            return coerce_bool(data.get("answer", False))
        except Exception as e:
            logger.warning(f"Grok call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(retry_sleep_s * attempt)
            else:
                raise
    return False


def find_col(fieldnames: List[str], candidates: List[str]) -> str:
    cand_lower = [c.strip().lower() for c in candidates]
    for c in fieldnames:
        if str(c).strip().lower() in cand_lower:
            return c
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="Path to note .txt")
    ap.add_argument("--infile", default="final_terms.csv", help="Path to input CSV/TSV")
    ap.add_argument("--outfile", default="final_terms_categorized.csv", help="Path to output CSV/TSV")
    ap.add_argument("--model", default="grok-4-1-fast-reasoning", help="xAI model name")
    ap.add_argument("--cache", default="grok_categorize_cache.json", help="JSON cache file")
    ap.add_argument("--log", default="categorize_terms.log", help="Log file path")
    ap.add_argument("--verbose", action="store_true", help="Verbose console logging")
    ap.add_argument("--sleep-s", type=float, default=0.0, help="Sleep between Grok calls")
    args = ap.parse_args()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise SystemExit("Set XAI_API_KEY in your environment (export XAI_API_KEY='xai-...').")
    logger = setup_logger(args.log, args.verbose)
    logger.info("Starting categorization run")
    logger.info(f"Input: {args.infile} | Note: {args.note} | Output: {args.outfile}")
    logger.info(f"Model: {args.model} | Cache: {args.cache} | Log: {args.log}")

    with open(args.note, "r", encoding="utf-8") as f:
        note_text = f.read()

    note_id = note_sha1(note_text)
    delimiter = sniff_delimiter(args.infile)
    rows, fieldnames = read_table(args.infile, delimiter)
    logger.info(f"Detected delimiter: {'TAB' if delimiter == chr(9) else 'COMMA'}")
    logger.info(f"Loaded {len(rows)} rows")

    hpo_col = find_col(fieldnames, ["HPO ID", "HPO_ID", "HPOID", "Canonical_ID"])
    term_col = find_col(fieldnames, ["Term Name", "Term_Name", "Name"])
    official_col = find_col(fieldnames, ["Official_Term_Name", "Official Term Name", "OfficialTermName"])

    if not term_col and not official_col:
        raise SystemExit("Input file missing required column: Term Name or Official_Term_Name")

    unique_terms: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for r in rows:
        term_name = normalize_term(r.get(term_col, "")) if term_col else ""
        official_name = normalize_term(r.get(official_col, "")) if official_col else ""
        if not term_name and official_name:
            term_name = official_name
        if not term_name:
            continue
        hpo_id = normalize_term(r.get(hpo_col, "")) if hpo_col else ""
        key = hpo_id or term_name.lower()
        if key not in unique_terms:
            unique_terms[key] = {
                "HPO ID": hpo_id,
                "Term Name": term_name,
                "Official_Term_Name": set(),
            }
        if official_name:
            unique_terms[key]["Official_Term_Name"].add(official_name)
        else:
            unique_terms[key]["Official_Term_Name"].add(term_name)

    logger.info(f"Unique terms: {len(unique_terms)}")

    cache = load_cache(args.cache)
    note_cache = cache.get(note_id, {})
    if not isinstance(note_cache, dict):
        note_cache = {}

    client = Client(api_key=api_key)
    system_prompt = build_system_prompt()

    questions = OrderedDict(
        [
            (
                "Inferred",
                "Is the phenotype inferred from objective data in the note and NOT explicitly stated anywhere?",
            ),
            (
                "Active",
                "Is the phenotype a current/active problem for the patient (not history/resolved)?",
            ),
            (
                "HPI",
                "Based on the note, is the phenotype present or clearly inferred within the HPI (History of Present Illness) section?",
            ),
            (
                "History",
                "Based on the note, is the phenotype present or clearly inferred within any history section (PMH, birth, developmental, family, social) or Review of Systems (ROS)?",
            ),
            (
                "Exam",
                "Based on the note, is the phenotype present or clearly inferred within the physical exam section?",
            ),
            (
                "Supporting",
                "Based on the note, is the phenotype present or clearly inferred within supporting data (labs, imaging, diagnostics)?",
            ),
            (
                "A/P",
                "Based on the note, is the phenotype present or clearly inferred within the Assessment paragraph, Plan, or Diagnosis/Problem List section?",
            ),
        ]
    )

    output_rows: List[Dict[str, Any]] = []
    for item in unique_terms.values():
        term_name = item["Term Name"]
        hpo_id = item["HPO ID"]
        official_terms = "; ".join(sorted(item.get("Official_Term_Name", set())))
        term_cache = note_cache.get(term_name, {})
        if not isinstance(term_cache, dict):
            term_cache = {}

        result_row = {
            "HPO ID": hpo_id,
            "Term Name": term_name,
            "Official_Term_Name": official_terms,
        }
        for col_name, question in questions.items():
            cache_key = col_name.lower().replace("/", "_")
            if cache_key in term_cache:
                answer = coerce_bool(term_cache.get(cache_key))
            else:
                logger.info(f"Grok check: {term_name} | {col_name}")
                answer = grok_yes_no(
                    logger=logger,
                    client=client,
                    model=args.model,
                    system_prompt=system_prompt,
                    term=term_name,
                    official_terms=official_terms,
                    hpo_id=hpo_id,
                    question=question,
                    note_text=note_text,
                )
                term_cache[cache_key] = bool(answer)
                note_cache[term_name] = term_cache
                cache.setdefault(note_id, {})
                cache[note_id] = note_cache
                save_cache(args.cache, cache)
                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)

            result_row[col_name] = "TRUE" if answer else "FALSE"

        output_rows.append(result_row)

    out_fields = ["HPO ID", "Term Name", "Official_Term_Name"] + list(questions.keys())
    write_table(args.outfile, delimiter, out_fields, output_rows)
    logger.info(f"Wrote output: {args.outfile}")
    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())