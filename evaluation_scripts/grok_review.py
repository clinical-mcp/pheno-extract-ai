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

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")


def setup_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("present_in_note")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Avoid duplicate handlers if re-imported
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


def normalize_term(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"\s*\([^)]*\)", "", s)  # remove parenthetical content
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def extract_family_history_block(note_text: str) -> str:
    lines = note_text.splitlines()
    start_idx = None
    end_idx = None

    fam_start_re = re.compile(r"^\s*(family history|fam(?:ily)?\s*hx|fhx)\b\s*:?\s*$", re.I)
    stop_re = re.compile(
        r"^\s*(hpi|history of present illness|pmh|past medical history|"
        r"social history|shx|ros|review of systems|physical exam|exam|"
        r"assessment|plan|a/p|medications|allergies|labs|imaging|diagnosis|"
        r"birth history|pregnancy history|developmental history)\b\s*:?\s*$",
        re.I,
    )

    for i, ln in enumerate(lines):
        if start_idx is None and fam_start_re.match(ln.strip()):
            start_idx = i
            continue
        if start_idx is not None and i > start_idx:
            if stop_re.match(ln.strip()) and not fam_start_re.match(ln.strip()):
                end_idx = i
                break

    if start_idx is None:
        return ""
    if end_idx is None:
        end_idx = len(lines)

    block = "\n".join(lines[start_idx:end_idx]).strip()
    return block if len(block) >= 30 else ""


def build_system_prompt() -> str:
    return (
        "You are an expert pediatric genetics chart reviewer.\n"
        "Given a clinical note and a list of phenotype terms, decide whether EACH term is present IN THE PATIENT.\n\n"
        "Very important rules:\n"
        "1) Mentions in FAMILY HISTORY (relatives) do NOT count as present in the patient unless the note explicitly "
        "states the patient has the finding.\n"
        "2) NEGATED mentions do NOT count. Examples: 'no seizures', 'denies seizures', 'negative for seizures', "
        "'without seizures', 'rule out seizures', 'seizures were not present'.\n"
        "3) Historical uncertainty does NOT count unless affirmed. Examples: 'possible seizures' or 'concern for seizures' "
        "should usually be false unless later confirmed.\n\n"
        "You MAY mark a term present if supported by:\n"
        "  (a) explicit patient-positive wording in the note (DIRECT), OR\n"
        "  (b) objective data in the note that meets standard clinical definitions (INFERRED).\n\n"
        "DIRECT vs INFERRED classification:\n"
        "- support='direct' ONLY if the note explicitly states the patient has the term (or an unambiguous synonym), e.g., "
        "'patient has microcephaly', 'history of hypothyroidism'.\n"
        "- support='inferred' ONLY if the term is NOT explicitly stated, but objective data supports it, e.g., "
        "'OFC 1st percentile' for microcephaly; 'TSH high and free T4 low' for hypothyroidism.\n"
        "- If BOTH direct and inferred evidence exist, choose support='direct'.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  \"<term>\": {\"present\": true/false, \"support\": \"direct\"/\"inferred\"/\"\", \"evidence\": \"<ONE line/sentence copied verbatim from the note>\"},\n'
        "  ...\n"
        "}\n\n"
        "Evidence rules:\n"
        "- If present=true, evidence MUST be one verbatim line/sentence from the note that supports a PATIENT-POSITIVE finding.\n"
        "- If present=true and support='direct', evidence should be the explicit patient-positive statement if available.\n"
        "- If present=true and support='inferred', evidence should contain the objective data used for inference.\n"
        "- If present=false, support must be \"\" and evidence must be \"\".\n"
        "- Keys must exactly match the provided term strings.\n"
        "- No extra keys. No commentary. JSON only.\n\n"
        "Helpful objective thresholds/examples (use when the note provides objective data):\n"
        "- Short stature: height z-score <= -2 OR height < 3rd percentile.\n"
        "- Tall stature: height z-score >= +2 OR height > 97th percentile.\n"
        "- Microcephaly: head circumference z-score <= -2 OR HC < 3rd percentile.\n"
        "- Macrocephaly: head circumference z-score >= +2 OR HC > 97th percentile.\n"
        "- Underweight / poor weight gain: weight-for-age z-score <= -2 OR weight < ~3rd–5th percentile.\n"
        "If the note provides borderline values or unclear context, prefer present=false."
    )


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


def _normalize_support(value: Any) -> str:
    s = (value or "").strip().lower()
    if s in ("direct", "inferred"):
        return s
    return ""


def grok_batch_check(
    logger: logging.Logger,
    client: Client,
    model: str,
    system_prompt: str,
    note_text: str,
    family_history_block: str,
    terms: List[str],
    max_retries: int = 4,
    retry_sleep_s: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Grok call: {len(terms)} terms (attempt {attempt}/{max_retries})")
            chat = client.chat.create(model=model)
            chat.append(system(system_prompt))

            user_prompt = (
                "TERMS (JSON array):\n"
                f"{json.dumps(terms, ensure_ascii=False)}\n\n"
                "CLINICAL NOTE:\n"
                "-----\n"
                f"{note_text}\n"
                "-----\n\n"
                "EXTRACTED FAMILY HISTORY (for exclusion only; do NOT count these as patient findings):\n"
                "-----\n"
                f"{family_history_block}\n"
                "-----\n\n"
                "Return JSON exactly per schema."
            )
            chat.append(user(user_prompt))
            response = chat.sample()
            data = extract_first_json_object(response.content)

            out: Dict[str, Dict[str, Any]] = {}
            for t in terms:
                v = data.get(t, {})
                if isinstance(v, dict):
                    present = bool(v.get("present", False))
                    evidence = str(v.get("evidence", "") or "")
                    support = _normalize_support(v.get("support", ""))

                    if not present:
                        evidence = ""
                        support = ""
                    else:
                        # If present=true but support invalid/empty, choose best-effort default.
                        if support not in ("direct", "inferred"):
                            support = "direct" if evidence else "inferred"
                    out[t] = {"present": present, "support": support, "evidence": evidence}
                else:
                    out[t] = {"present": False, "support": "", "evidence": ""}

            # Log a compact preview
            preview_true = [t for t in terms if out.get(t, {}).get("present") is True]
            logger.info(f"  -> present=true for {len(preview_true)}/{len(terms)} terms")
            if preview_true:
                for t in preview_true[:5]:
                    ev = out[t].get("evidence", "")
                    sup = out[t].get("support", "")
                    logger.debug(f"     TRUE: {t} | support: {sup} | evidence: {ev[:200]}")
                if len(preview_true) > 5:
                    logger.debug(f"     ... ({len(preview_true)-5} more true terms in this batch)")
            return out

        except Exception as e:
            logger.warning(f"Grok batch failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(retry_sleep_s * attempt)
            else:
                raise

    return {t: {"present": False, "support": "", "evidence": ""} for t in terms}


def batched(items: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="Path to note .txt")
    ap.add_argument("--infile", required=True, help="Path to input CSV/TSV")
    ap.add_argument("--outfile", required=True, help="Path to output CSV/TSV")
    ap.add_argument("--model", default="grok-4-1-fast-reasoning", help="xAI model name")
    ap.add_argument("--batch-size", type=int, default=25, help="Unique normalized terms per Grok call")
    ap.add_argument("--cache", default="grok_presence_cache.json", help="JSON cache file (note-specific)")
    ap.add_argument("--evidence-col", default="MANUAL_Present_In_Note_Evidence", help="Evidence column name")
    ap.add_argument(
        "--support-col",
        default="MANUAL_Present_In_Note_Support",
        help="Support column name (direct|inferred when present=true)",
    )
    ap.add_argument("--log", default="present_in_note.log", help="Log file path")
    ap.add_argument("--verbose", action="store_true", help="Verbose console logging")
    args = ap.parse_args()

    api_key = XAI_API_KEY
    if not api_key:
        raise SystemExit("Set XAI_API_KEY in your environment (export XAI_API_KEY='xai-...').")

    logger = setup_logger(args.log, args.verbose)
    logger.info("Starting annotation run")
    logger.info(f"Input: {args.infile} | Note: {args.note} | Output: {args.outfile}")
    logger.info(
        f"Model: {args.model} | Batch size: {args.batch_size} | Cache: {args.cache} | Log: {args.log}"
    )

    with open(args.note, "r", encoding="utf-8") as f:
        note_text = f.read()

    note_id = note_sha1(note_text)
    family_history_block = extract_family_history_block(note_text)
    logger.debug(f"NoteID: {note_id}")
    logger.debug(f"Family history block chars: {len(family_history_block)}")

    delimiter = sniff_delimiter(args.infile)
    rows, fieldnames = read_table(args.infile, delimiter)
    logger.info(f"Detected delimiter: {'TAB' if delimiter == chr(9) else 'COMMA'}")
    logger.info(f"Loaded {len(rows)} rows")

    if "Raw_Term" not in fieldnames:
        raise SystemExit("Input file missing required column: Raw_Term")

    present_col = "MANUAL_Present_In_Note"
    evidence_col = args.evidence_col
    support_col = args.support_col

    if present_col not in fieldnames:
        fieldnames.append(present_col)
        logger.info(f"Added column: {present_col}")
    if evidence_col not in fieldnames:
        fieldnames.append(evidence_col)
        logger.info(f"Added column: {evidence_col}")
    if support_col not in fieldnames:
        fieldnames.append(support_col)
        logger.info(f"Added column: {support_col}")

    unique_norm_terms = OrderedDict()
    empty_norm = 0
    for r in rows:
        norm = normalize_term(r.get("Raw_Term", ""))
        if norm:
            unique_norm_terms.setdefault(norm, True)
        else:
            empty_norm += 1
    logger.info(f"Unique normalized terms: {len(unique_norm_terms)} (empty Raw_Term rows: {empty_norm})")

    cache = load_cache(args.cache)
    note_cache = cache.get(note_id, {})
    if not isinstance(note_cache, dict):
        note_cache = {}

    results: Dict[str, Dict[str, Any]] = {}
    for k, v in note_cache.items():
        if isinstance(v, dict):
            present = bool(v.get("present", False))
            evidence = str(v.get("evidence", "") or "")
            support = _normalize_support(v.get("support", ""))
            if not present:
                evidence = ""
                support = ""
            results[str(k)] = {"present": present, "support": support, "evidence": evidence}
    logger.info(f"Cache hits for this note: {len(results)}")

    to_ask = [t for t in unique_norm_terms.keys() if t not in results]
    logger.info(f"Terms needing Grok: {len(to_ask)}")

    client = Client(api_key=api_key)
    system_prompt = build_system_prompt()

    batches = batched(to_ask, args.batch_size)
    logger.info(f"Total Grok calls to make: {len(batches)}")

    for i, batch in enumerate(batches, start=1):
        logger.info(f"Batch {i}/{len(batches)}")
        batch_res = grok_batch_check(
            logger=logger,
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            note_text=note_text,
            family_history_block=family_history_block,
            terms=batch,
        )
        results.update(batch_res)

    cache.setdefault(note_id, {})
    for term, val in results.items():
        cache[note_id][term] = {
            "present": bool(val.get("present", False)),
            "support": _normalize_support(val.get("support", "")),
            "evidence": val.get("evidence", "") or "",
        }
    save_cache(args.cache, cache)
    logger.info("Cache updated")

    true_count = 0
    for r in rows:
        norm = normalize_term(r.get("Raw_Term", ""))
        if not norm:
            continue
        hit = results.get(norm, {"present": False, "support": "", "evidence": ""})
        is_true = bool(hit.get("present", False))
        r[present_col] = "TRUE" if is_true else "FALSE"
        r[evidence_col] = hit.get("evidence", "") or "" if is_true else ""
        r[support_col] = hit.get("support", "") or "" if is_true else ""
        if is_true:
            true_count += 1

    write_table(args.outfile, delimiter, fieldnames, rows)
    logger.info(f"Wrote output: {args.outfile}")
    logger.info(f"Rows marked TRUE: {true_count}/{len(rows)}")
    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




def setup_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("present_in_note")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Avoid duplicate handlers if re-imported
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


def normalize_term(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"\s*\([^)]*\)", "", s)  # remove parenthetical content
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def extract_family_history_block(note_text: str) -> str:
    lines = note_text.splitlines()
    start_idx = None
    end_idx = None

    fam_start_re = re.compile(r"^\s*(family history|fam(?:ily)?\s*hx|fhx)\b\s*:?\s*$", re.I)
    stop_re = re.compile(
        r"^\s*(hpi|history of present illness|pmh|past medical history|"
        r"social history|shx|ros|review of systems|physical exam|exam|"
        r"assessment|plan|a/p|medications|allergies|labs|imaging|diagnosis|"
        r"birth history|pregnancy history|developmental history)\b\s*:?\s*$",
        re.I,
    )

    for i, ln in enumerate(lines):
        if start_idx is None and fam_start_re.match(ln.strip()):
            start_idx = i
            continue
        if start_idx is not None and i > start_idx:
            if stop_re.match(ln.strip()) and not fam_start_re.match(ln.strip()):
                end_idx = i
                break

    if start_idx is None:
        return ""
    if end_idx is None:
        end_idx = len(lines)

    block = "\n".join(lines[start_idx:end_idx]).strip()
    return block if len(block) >= 30 else ""


def build_system_prompt() -> str:
    return (
        "You are an expert pediatric genetics chart reviewer.\n"
        "Given a clinical note and a list of phenotype terms, decide whether EACH term is present IN THE PATIENT.\n\n"
        "Very important rules:\n"
        "1) Mentions in FAMILY HISTORY (relatives) do NOT count as present in the patient unless the note explicitly "
        "states the patient has the finding.\n"
        "2) NEGATED mentions do NOT count. Examples: 'no seizures', 'denies seizures', 'negative for seizures', "
        "'without seizures', 'rule out seizures', 'seizures were not present'.\n"
        "3) Historical uncertainty does NOT count unless affirmed. Examples: 'possible seizures' or 'concern for seizures' "
        "should usually be false unless later confirmed.\n\n"
        "You MAY mark a term present if supported by:\n"
        "  (a) explicit patient-positive wording in the note, OR\n"
        "  (b) objective data in the note that meets standard clinical definitions (percentiles, z-scores, numeric values).\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  \"<term>\": {\"present\": true/false, \"evidence\": \"<ONE line/sentence copied verbatim from the note>\"},\n'
        "  ...\n"
        "}\n\n"
        "Evidence rules:\n"
        "- If present=true, evidence MUST be one verbatim line/sentence from the note that supports a PATIENT-POSITIVE finding.\n"
        "- If present=false, evidence must be \"\".\n"
        "- Keys must exactly match the provided term strings.\n"
        "- No extra keys. No commentary. JSON only.\n\n"
        "Helpful objective thresholds/examples (use when the note provides objective data):\n"
        "- Short stature: height z-score <= -2 OR height < 3rd percentile.\n"
        "- Tall stature: height z-score >= +2 OR height > 97th percentile.\n"
        "- Microcephaly: head circumference z-score <= -2 OR HC < 3rd percentile.\n"
        "- Macrocephaly: head circumference z-score >= +2 OR HC > 97th percentile.\n"
        "- Underweight / poor weight gain: weight-for-age z-score <= -2 OR weight < ~3rd–5th percentile.\n"
        "If the note provides borderline values or unclear context, prefer present=false."
    )


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


def grok_batch_check(
    logger: logging.Logger,
    client: Client,
    model: str,
    system_prompt: str,
    note_text: str,
    family_history_block: str,
    terms: List[str],
    max_retries: int = 4,
    retry_sleep_s: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Grok call: {len(terms)} terms (attempt {attempt}/{max_retries})")
            chat = client.chat.create(model=model)
            chat.append(system(system_prompt))

            user_prompt = (
                "TERMS (JSON array):\n"
                f"{json.dumps(terms, ensure_ascii=False)}\n\n"
                "CLINICAL NOTE:\n"
                "-----\n"
                f"{note_text}\n"
                "-----\n\n"
                "EXTRACTED FAMILY HISTORY (for exclusion only; do NOT count these as patient findings):\n"
                "-----\n"
                f"{family_history_block}\n"
                "-----\n\n"
                "Return JSON exactly per schema."
            )
            chat.append(user(user_prompt))
            response = chat.sample()
            data = extract_first_json_object(response.content)

            out: Dict[str, Dict[str, Any]] = {}
            for t in terms:
                v = data.get(t, {})
                if isinstance(v, dict):
                    present = bool(v.get("present", False))
                    evidence = v.get("evidence", "") or ""
                    if not present:
                        evidence = ""
                    out[t] = {"present": present, "evidence": str(evidence)}
                else:
                    out[t] = {"present": False, "evidence": ""}

            # Log a compact preview
            preview_true = [t for t in terms if out.get(t, {}).get("present") is True]
            logger.info(f"  -> present=true for {len(preview_true)}/{len(terms)} terms")
            if preview_true:
                for t in preview_true[:5]:
                    ev = out[t].get("evidence", "")
                    logger.debug(f"     TRUE: {t} | evidence: {ev[:200]}")
                if len(preview_true) > 5:
                    logger.debug(f"     ... ({len(preview_true)-5} more true terms in this batch)")
            return out

        except Exception as e:
            logger.warning(f"Grok batch failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(retry_sleep_s * attempt)
            else:
                raise

    return {t: {"present": False, "evidence": ""} for t in terms}


def batched(items: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="Path to note .txt")
    ap.add_argument("--infile", required=True, help="Path to input CSV/TSV")
    ap.add_argument("--outfile", required=True, help="Path to output CSV/TSV")
    ap.add_argument("--model", default="grok-4-1-fast-reasoning", help="xAI model name")
    ap.add_argument("--batch-size", type=int, default=25, help="Unique normalized terms per Grok call")
    ap.add_argument("--cache", default="grok_presence_cache.json", help="JSON cache file (note-specific)")
    ap.add_argument("--evidence-col", default="MANUAL_Present_In_Note_Evidence", help="Evidence column name")
    ap.add_argument("--log", default="present_in_note.log", help="Log file path")
    ap.add_argument("--verbose", action="store_true", help="Verbose console logging")
    args = ap.parse_args()

    api_key = XAI_API_KEY
    if not api_key:
        raise SystemExit("Set XAI_API_KEY in your environment (export XAI_API_KEY='xai-...').")

    logger = setup_logger(args.log, args.verbose)
    logger.info("Starting annotation run")
    logger.info(f"Input: {args.infile} | Note: {args.note} | Output: {args.outfile}")
    logger.info(f"Model: {args.model} | Batch size: {args.batch_size} | Cache: {args.cache} | Log: {args.log}")

    with open(args.note, "r", encoding="utf-8") as f:
        note_text = f.read()

    note_id = note_sha1(note_text)
    family_history_block = extract_family_history_block(note_text)
    logger.debug(f"NoteID: {note_id}")
    logger.debug(f"Family history block chars: {len(family_history_block)}")

    delimiter = sniff_delimiter(args.infile)
    rows, fieldnames = read_table(args.infile, delimiter)
    logger.info(f"Detected delimiter: {'TAB' if delimiter == chr(9) else 'COMMA'}")
    logger.info(f"Loaded {len(rows)} rows")

    if "Raw_Term" not in fieldnames:
        raise SystemExit("Input file missing required column: Raw_Term")

    present_col = "MANUAL_Present_In_Note"
    evidence_col = args.evidence_col

    if present_col not in fieldnames:
        fieldnames.append(present_col)
        logger.info(f"Added column: {present_col}")
    if evidence_col not in fieldnames:
        fieldnames.append(evidence_col)
        logger.info(f"Added column: {evidence_col}")

    unique_norm_terms = OrderedDict()
    empty_norm = 0
    for r in rows:
        norm = normalize_term(r.get("Raw_Term", ""))
        if norm:
            unique_norm_terms.setdefault(norm, True)
        else:
            empty_norm += 1
    logger.info(f"Unique normalized terms: {len(unique_norm_terms)} (empty Raw_Term rows: {empty_norm})")

    cache = load_cache(args.cache)
    note_cache = cache.get(note_id, {})
    if not isinstance(note_cache, dict):
        note_cache = {}

    results: Dict[str, Dict[str, Any]] = {}
    for k, v in note_cache.items():
        if isinstance(v, dict):
            results[str(k)] = {
                "present": bool(v.get("present", False)),
                "evidence": str(v.get("evidence", "") or ""),
            }
    logger.info(f"Cache hits for this note: {len(results)}")

    to_ask = [t for t in unique_norm_terms.keys() if t not in results]
    logger.info(f"Terms needing Grok: {len(to_ask)}")

    client = Client(api_key=api_key)
    system_prompt = build_system_prompt()

    batches = batched(to_ask, args.batch_size)
    logger.info(f"Total Grok calls to make: {len(batches)}")

    for i, batch in enumerate(batches, start=1):
        logger.info(f"Batch {i}/{len(batches)}")
        batch_res = grok_batch_check(
            logger=logger,
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            note_text=note_text,
            family_history_block=family_history_block,
            terms=batch,
        )
        results.update(batch_res)

    cache.setdefault(note_id, {})
    for term, val in results.items():
        cache[note_id][term] = {
            "present": bool(val.get("present", False)),
            "evidence": val.get("evidence", "") or "",
        }
    save_cache(args.cache, cache)
    logger.info("Cache updated")

    true_count = 0
    for r in rows:
        norm = normalize_term(r.get("Raw_Term", ""))
        if not norm:
            continue
        hit = results.get(norm, {"present": False, "evidence": ""})
        is_true = bool(hit.get("present", False))
        r[present_col] = "TRUE" if is_true else "FALSE"
        r[evidence_col] = hit.get("evidence", "") or ""
        if is_true:
            true_count += 1

    write_table(args.outfile, delimiter, fieldnames, rows)
    logger.info(f"Wrote output: {args.outfile}")
    logger.info(f"Rows marked TRUE: {true_count}/{len(rows)}")
    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#py present_in_note.py --note /path/to/note.txt  --infile /path/to/terms.csv  --outfile /path/to/terms_annotated.csv  --model grok-4-1-fast-reasoning --batch-size 25 --cache grok_presence_cache.json --evidence-col MANUAL_Present_In_Note_Evidence --support-col MANUAL_Present_In_Note_Support --log present_in_note.log --verbose