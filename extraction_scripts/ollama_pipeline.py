import os

# Optional: ignore proxies for LAN + loopback
os.environ["NO_PROXY"] = "192.168.50.111,localhost,127.0.0.1"
os.environ["no_proxy"] = "192.168.50.111,localhost,127.0.0.1"

import glob
import re
import csv
import json
import sys
import time
import threading
import httpx
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional


# =====================
# CONFIGURATION
# =====================

# Ollama host (Unraid)
OLLAMA_SERVER_URL = "http://192.168.50.111:11434"

# Model roles
EXTRACTOR_MODEL = "deepseek-r1:14b"   # reasoning/extraction
MAPPER_MODEL = "qwen2.5:14b"         # MCP/tools fallback mapper

# MCP (your HPO lookup server exposed via ngrok SSE)
MCP_SERVER_URL = "https://postsigmoidal-rosanne-interportal.ngrok-free.dev/sse"

HPO_JSON_FILE = "hp.json"

NUM_RUNS = 2

# Per-request timeout (seconds). If exceeded, that run continues and checkpoint advances.
OLLAMA_REQUEST_TIMEOUT_S = 240

# VRAM unload between calls (slower but avoids “everything stays loaded”)
KEEP_ALIVE_EXTRACTOR = "0s"
KEEP_ALIVE_MAPPER = "0s"

# Deterministic-ish
EXTRACTOR_OPTIONS = {
    "temperature": 0,
    "num_ctx": 16384,     # raise if model supports it
    "num_predict": 1200,  # room to list many findings
    "seed": 1,
}
MAPPER_OPTIONS = {"temperature": 0, "top_k": 1, "seed": 1}

# Local fuzzy remap threshold (0..1). Higher = stricter / fewer false matches
FUZZY_MIN_SCORE = 0.9

# If Qwen fails to output a parseable [HP:...] line, fallback to MCP top result directly
USE_MCP_TOP1_IF_QWEN_FAILS = True

# Tool loop limit
MAX_TOOL_ROUNDS = 4

# Chunk size for extractor to avoid context truncation
EXTRACTOR_CHUNK_CHARS = 12000


# =====================
# PROMPTS
# =====================

# Comparable to your other models: DeepSeek only extracts findings, no IDs, no tool talk.
# Key: explicitly disallow summarizing.
EXTRACTOR_SYSTEM_PROMPT = """
You are an expert Clinical Geneticist and phenotype extractor.

Goal: Produce a COMPREHENSIVE but ACCURATE list of patient phenotypic abnormalities from this note.
Do NOT invent findings. Only include a finding if it is clearly supported in the note (explicitly stated or confidently inferable from objective data).

Critical inclusion/exclusion rules:
- Count ONLY findings in the PATIENT. Family history does NOT count unless the note explicitly states the patient has the finding.
- Exclude negated findings (e.g., “no seizures”, “denies seizures”, “negative for …”).
- Exclude uncertainty (“possible”, “concern for”, “rule out”) unless later confirmed.
- Do NOT add “expected” features of a suspected diagnosis unless documented in the patient.
- Avoid duplicative near-synonyms; prefer the most appropriate concepts.

Inference rules (high recall, controlled):
You MAY infer a phenotype ONLY when objective data meet common clinical thresholds/patterns:
- Microcephaly: OFC/HC < 3rd percentile or z ≤ -2
- Macrocephaly: OFC/HC > 97th percentile or z ≥ +2
- Short stature: height < 3rd percentile or z ≤ -2
- Tall stature: height > 97th percentile or z ≥ +2
- Underweight/failure to thrive: weight < ~3rd–5th percentile or z ≤ -2, or clear longitudinal concern
- Anemia: low hemoglobin/hematocrit; specify subtype only if stated or clearly supported
- Hypothyroidism: elevated TSH with low free T4 (or explicit diagnosis)

IMPORTANT:
- Do NOT summarize or “pick the most important” findings. Include ALL abnormalities you can support.

Output format (ONLY):
- One bullet per finding (plain text), no HPO IDs.
"""

# Qwen handles unresolved items, MUST use the tool.
MAPPER_ONE_SYSTEM_PROMPT = """
You are a Medical Terminology Mapper.

Task: Map ONE clinical finding to the single best official HPO term.

You MUST call `search_hpo_terms` with an appropriate phenotype phrase for this finding,
then choose the best match from the returned candidates.

Output format (ONLY):
- [HP:########] Official Term Name

If you truly cannot find a match, output exactly:
NO_MATCH
"""


# =====================
# Normalization / matching (post-processing vs hp.json)
# =====================

_ROMAN_TYPE_MAP = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}

STOPWORDS = {
    "of", "the", "and", "or", "to", "due", "with", "without", "in", "on", "a", "an", "for", "from",
    "border",
}

TOKEN_CANON = {
    "vermillion": "vermilion",
    "inequality": "discrepancy",
    "inequalities": "discrepancy",
    "unequal": "discrepancy",
    "leg": "limb",
    "aversion": "anteversion",   # common LLM slip
    "varus": "vara",             # can help in some HPO strings
}

def strip_think_blocks(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()

def remove_parentheticals(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s*\([^)]*\)", "", s).strip()

def normalize_text(s: str) -> str:
    if not s:
        return ""

    s = remove_parentheticals(s)
    s = s.strip().lower()

    # normalize punctuation to spaces
    s = re.sub(r"[/,_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # normalize "type i" -> "type 1" etc.
    def _type_roman_to_num(m):
        roman = m.group(1).lower()
        return "type " + _ROMAN_TYPE_MAP.get(roman, roman)

    s = re.sub(r"\btype\s+(i|ii|iii|iv|v|vi)\b", _type_roman_to_num, s)

    # reorder "malformation type 1" -> "type 1 malformation"
    s = re.sub(r"\bmalformation\s+type\s+(\d+)\b", r"type \1 malformation", s)

    # reorder "chiari malformation type 1" -> "chiari type 1 malformation"
    s = re.sub(r"\bchiari\s+malformation\s+type\s+(\d+)\b", r"chiari type \1 malformation", s)

    # common shorthand
    s = s.replace("chiari i", "chiari type 1")
    s = s.replace("chiari 1", "chiari type 1")

    # phrase-level nudge
    s = re.sub(r"\bleg\s+length\s+inequalit(y|ies)\b", "limb length discrepancy", s)
    s = re.sub(r"\binequalit(y|ies)\s+in\s+leg\s+length\b", "limb length discrepancy", s)

    s = re.sub(r"[;:,\.\s]+$", "", s).strip()
    return s

def bow_key(s: str) -> str:
    s = normalize_text(s)
    if not s:
        return ""
    toks = []
    for t in s.split():
        if t in STOPWORDS:
            continue
        t = TOKEN_CANON.get(t, t)
        toks.append(t)
    if not toks:
        return ""
    toks = sorted(set(toks))
    return " ".join(toks)

def normalize_hpo_id(hpo_id: str) -> str:
    if not hpo_id:
        return ""
    hpo_id = hpo_id.strip()
    if hpo_id.upper().startswith("HP_"):
        return "HP:" + hpo_id.split("HP_")[-1]
    if hpo_id.upper().startswith("HP:"):
        return "HP:" + hpo_id.split("HP:")[-1]
    return hpo_id

def get_term_id(term: Dict) -> str:
    tid = (term.get("id") or "").strip()
    return normalize_hpo_id(tid)

def get_term_label(term: Dict) -> str:
    return (term.get("lbl") or term.get("name") or "").strip()

def get_term_synonyms(term: Dict) -> List[str]:
    meta = term.get("meta", {}) or {}
    synonyms_list = meta.get("synonyms", []) or []
    out = []
    for s in synonyms_list:
        if isinstance(s, dict):
            v = (s.get("val") or "").strip()
            if v:
                out.append(v)
        elif isinstance(s, str):
            v = s.strip()
            if v:
                out.append(v)
    return out

def jaccard_tokens(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    A = set(a.split())
    B = set(b.split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

# Globals built from hp.json
HPO_NODES: List[Dict] = []
VALID_HPO_IDS: Set[str] = set()
ID_TO_OFFICIAL: Dict[str, str] = {}
NAME_NORM_TO_ID: Dict[str, str] = {}
SYN_NORM_TO_IDS: Dict[str, List[str]] = {}
BOW_KEY_TO_IDS: Dict[str, List[str]] = {}
TOKEN_INDEX: Dict[str, Set[str]] = {}          # token -> set(norm_string)
NORM_STRING_TO_IDS: Dict[str, List[str]] = {}  # norm string -> ids

def load_hpo_db(filepath: str):
    global HPO_NODES, VALID_HPO_IDS, ID_TO_OFFICIAL, NAME_NORM_TO_ID, SYN_NORM_TO_IDS
    global BOW_KEY_TO_IDS, TOKEN_INDEX, NORM_STRING_TO_IDS

    print(f"Loading HPO data from {filepath} for validation/remap...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "graphs" in data:
            HPO_NODES = data["graphs"][0]["nodes"]
        elif isinstance(data, list):
            HPO_NODES = data
        else:
            HPO_NODES = data

        VALID_HPO_IDS = set()
        ID_TO_OFFICIAL = {}
        NAME_NORM_TO_ID = {}
        SYN_NORM_TO_IDS = {}

        BOW_KEY_TO_IDS = {}
        TOKEN_INDEX = {}
        NORM_STRING_TO_IDS = {}

        for term in HPO_NODES:
            tid = get_term_id(term)
            name = get_term_label(term)
            if not tid or not name:
                continue

            VALID_HPO_IDS.add(tid)
            ID_TO_OFFICIAL[tid] = name

            name_norm = normalize_text(name)
            if name_norm:
                NAME_NORM_TO_ID.setdefault(name_norm, tid)

            for syn in get_term_synonyms(term):
                sn = normalize_text(syn)
                if not sn:
                    continue
                SYN_NORM_TO_IDS.setdefault(sn, []).append(tid)

        # exact string index
        for norm_name, tid in NAME_NORM_TO_ID.items():
            NORM_STRING_TO_IDS.setdefault(norm_name, []).append(tid)
        for norm_syn, tids in SYN_NORM_TO_IDS.items():
            for tid in tids:
                NORM_STRING_TO_IDS.setdefault(norm_syn, []).append(tid)

        # token index for fuzzy candidate generation
        for norm_str in NORM_STRING_TO_IDS.keys():
            toks = [t for t in norm_str.split() if len(t) > 2 and t not in STOPWORDS]
            for tok in toks:
                tok = TOKEN_CANON.get(tok, tok)
                TOKEN_INDEX.setdefault(tok, set()).add(norm_str)

        # bag-of-words exact index (official + synonyms)
        for tid, official in ID_TO_OFFICIAL.items():
            bk = bow_key(official)
            if bk:
                BOW_KEY_TO_IDS.setdefault(bk, []).append(tid)
        for syn_norm, tids in SYN_NORM_TO_IDS.items():
            bk = bow_key(syn_norm)
            if bk:
                for tid in tids:
                    BOW_KEY_TO_IDS.setdefault(bk, []).append(tid)

        print(f"SUCCESS: Loaded {len(VALID_HPO_IDS)} HPO terms.")
    except Exception as e:
        print(f"ERROR loading HPO JSON: {e}")
        sys.exit(1)

def fuzzy_find_best_id(name_norm: str, min_score: float) -> Tuple[str, str, float, str]:
    bk = bow_key(name_norm)
    if not bk:
        return "", "", 0.0, "no_candidates"

    toks = bk.split()
    candidates_norm_strings: Set[str] = set()
    for t in toks:
        for norm_str in TOKEN_INDEX.get(t, set()):
            candidates_norm_strings.add(norm_str)

    if not candidates_norm_strings:
        return "", "", 0.0, "no_candidates"

    best_id, best_name, best_score = "", "", 0.0
    for cand_norm in candidates_norm_strings:
        cand_bk = bow_key(cand_norm)
        score = jaccard_tokens(bk, cand_bk)
        if score > best_score:
            ids = NORM_STRING_TO_IDS.get(cand_norm, [])
            if ids:
                chosen = min(ids, key=lambda x: len(ID_TO_OFFICIAL.get(x, x)))
                best_id = chosen
                best_name = ID_TO_OFFICIAL.get(chosen, "")
                best_score = score

    if best_score >= min_score and best_id:
        return best_id, best_name, best_score, "fuzzy_jaccard"

    return "", "", best_score, "fuzzy_below_threshold"

def remap_against_hp_json(model_id: str, model_name: str, fuzzy_min_score: float) -> Tuple[str, str, str]:
    """
    1) exact official name
    2) exact synonym
    3) exact bag-of-words
    4) fuzzy fallback (Jaccard on normalized token sets)
    5) keep valid model_id if present
    """
    mid = normalize_hpo_id(model_id)
    name_clean = remove_parentheticals(model_name)
    name_norm = normalize_text(name_clean)

    if name_norm and name_norm in NAME_NORM_TO_ID:
        cid = NAME_NORM_TO_ID[name_norm]
        return cid, ID_TO_OFFICIAL.get(cid, ""), "name_exact"

    if name_norm and name_norm in SYN_NORM_TO_IDS:
        candidates = SYN_NORM_TO_IDS[name_norm]
        best_id = min(candidates, key=lambda x: len(ID_TO_OFFICIAL.get(x, x)))
        return best_id, ID_TO_OFFICIAL.get(best_id, ""), "syn_exact"

    bk = bow_key(name_clean)
    if bk and bk in BOW_KEY_TO_IDS:
        candidates = BOW_KEY_TO_IDS[bk]
        best_id = min(candidates, key=lambda x: len(ID_TO_OFFICIAL.get(x, x)))
        return best_id, ID_TO_OFFICIAL.get(best_id, ""), "bow_exact"

    if name_norm:
        fid, fname, score, via = fuzzy_find_best_id(name_norm, min_score=fuzzy_min_score)
        if fid and fname:
            return fid, fname, f"{via}:{score:.2f}"

    if mid and mid in VALID_HPO_IDS:
        return mid, ID_TO_OFFICIAL.get(mid, ""), "kept_valid_id"

    return "", "", "no_match"


# =====================
# MCP CLIENT (SSE transport compatible)
# =====================

class SimpleMCPClient:
    """
    FastMCP transport='sse' delivers JSON-RPC responses over the SSE stream, not in the POST body.
    This client:
      - keeps SSE open in a background thread
      - posts JSON-RPC requests to the endpoint
      - waits for the matching response id received via SSE
    """

    def __init__(self, sse_url: str):
        self.sse_url = sse_url
        self.post_url: Optional[str] = None
        self.tools = []

        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._next_id = 1
        self._cv = threading.Condition()
        self._responses = {}  # id -> json

    def _set_post_url(self, path: str):
        path = path.strip()
        base = self.sse_url.rsplit("/sse", 1)[0]
        self.post_url = path if path.startswith("http") else f"{base}{path}"

    def _sse_loop(self):
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        try:
            with self._http.stream("GET", self.sse_url, headers=headers) as resp:
                resp.raise_for_status()

                event_type = None
                data_lines = []

                for raw in resp.iter_lines():
                    if self._stop.is_set():
                        return
                    if raw is None:
                        continue

                    line = raw.strip()

                    # Blank line terminates an SSE event
                    if line == "":
                        if data_lines:
                            data = "\n".join(data_lines).strip()

                            if event_type == "endpoint":
                                self._set_post_url(data)
                            elif event_type is None and self.post_url is None and data:
                                self._set_post_url(data)
                            else:
                                try:
                                    msg = json.loads(data)
                                    if isinstance(msg, dict) and "id" in msg:
                                        with self._cv:
                                            self._responses[msg["id"]] = msg
                                            self._cv.notify_all()
                                except Exception:
                                    pass

                        event_type = None
                        data_lines = []
                        continue

                    if line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:"):].lstrip())
        except Exception:
            return

    def connect(self) -> bool:
        print("Connecting to MCP Server...", end=" ")
        self._thread = threading.Thread(target=self._sse_loop, daemon=True)
        self._thread.start()

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self.post_url:
                print("Connected.")
                return True
            time.sleep(0.05)

        print("Failed (no endpoint received from SSE).")
        return False

    def close(self):
        self._stop.set()
        try:
            self._http.close()
        except Exception:
            pass

    def _post(self, payload: dict):
        if not self.post_url:
            raise RuntimeError("MCP post_url not set (SSE endpoint handshake failed).")
        resp = self._http.post(self.post_url, json=payload)
        resp.raise_for_status()
        return resp

    def send_rpc(self, method: str, params=None, timeout: float = 60.0):
        with self._cv:
            req_id = self._next_id
            self._next_id += 1

        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": req_id}
        self._post(payload)

        deadline = time.time() + timeout
        with self._cv:
            while time.time() < deadline:
                if req_id in self._responses:
                    return self._responses.pop(req_id)
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=min(0.5, remaining))

        raise TimeoutError(f"MCP RPC timed out waiting for response id={req_id}, method={method}")

    def send_notification(self, method: str, params=None):
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._post(payload)

    def initialize(self):
        self.send_rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ollama-option-a", "version": "1.0"},
            },
        )
        self.send_notification("notifications/initialized", {})

    def list_tools(self):
        resp = self.send_rpc("tools/list")
        if "result" in resp and "tools" in resp["result"]:
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("inputSchema", {}),
                    },
                }
                for t in resp["result"]["tools"]
            ]
            return self.tools
        return []

    def call_tool(self, name, arguments):
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"_raw": arguments}

        resp = self.send_rpc("tools/call", {"name": name, "arguments": arguments})
        if "result" in resp:
            content = resp["result"].get("content", [])
            return "".join([item.get("text", "") for item in content if item.get("type") == "text"])
        return "Error."

    # helper: direct HPO search without LLM
    def search_hpo_terms(self, query: str) -> List[Dict]:
        txt = self.call_tool("search_hpo_terms", {"query": query})
        try:
            return json.loads(txt)
        except Exception:
            return []


# =====================
# Ollama HTTP client (explicit timeouts + keep_alive)
# =====================

class OllamaHTTP:
    def __init__(self, base_url: str, timeout_s: int):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=None))

    def close(self):
        try:
            self.http.close()
        except Exception:
            pass

    def chat(self, model: str, messages: List[Dict], tools: Optional[List[Dict]] = None,
             options: Optional[Dict] = None, keep_alive: Optional[str] = None) -> Dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = tools
        if options is not None:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        r = self.http.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


# =====================
# Parsing helpers
# =====================

def split_note_into_chunks(note: str, max_chars: int = 12000) -> List[str]:
    """
    Split on blank lines to keep clinical sections intact.
    max_chars is characters (rough proxy for tokens).
    """
    note = note or ""
    blocks = re.split(r"\n\s*\n", note.strip())
    chunks = []
    cur = []
    cur_len = 0

    for b in blocks:
        b = b.strip()
        if not b:
            continue
        add_len = len(b) + 2
        if cur and (cur_len + add_len > max_chars):
            chunks.append("\n\n".join(cur).strip())
            cur = [b]
            cur_len = len(b)
        else:
            cur.append(b)
            cur_len += add_len

    if cur:
        chunks.append("\n\n".join(cur).strip())

    return [c for c in chunks if c]

def parse_findings_list_more_tolerant(text: str) -> List[str]:
    """
    Tolerant of bullets, numbering, markdown bold, and extra intro lines.
    """
    text = strip_think_blocks(text)
    text = text.strip()
    if not text:
        return []

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # strip markdown bold
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)

        # strip bullets/numbering
        line = re.sub(r"^\s*[-*•]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        line = re.sub(r"^\s*\d+\)\s+", "", line)

        # remove trailing punctuation
        line = re.sub(r"[;:,\.\s]+$", "", line).strip()

        if not line:
            continue

        # ignore common preambles
        if line.lower().startswith(("based on", "here is", "these hpo", "output", "summary")):
            continue

        out.append(line)

    # dedupe by normalized text
    seen = set()
    deduped = []
    for f in out:
        k = normalize_text(f)
        if k and k not in seen:
            seen.add(k)
            deduped.append(f)
    return deduped

def deepseek_extract_findings_high_recall(
    ollama_http: OllamaHTTP,
    note: str,
    run_i: int,
    *,
    chunk_chars: int = 12000,
) -> Tuple[str, List[str]]:
    """
    Two-pass extraction:
      Pass 1: extract from each chunk
      Pass 2: delta sweep — for each chunk, ask for ONLY missing findings vs current list

    Returns:
      (extractor_debug_text, merged_findings)
    """
    chunks = split_note_into_chunks(note, max_chars=chunk_chars)

    all_findings: List[str] = []
    debug_parts = []

    def merge_findings(new_items: List[str]):
        nonlocal all_findings
        seen = {normalize_text(x) for x in all_findings}
        for it in new_items:
            k = normalize_text(it)
            if k and k not in seen:
                seen.add(k)
                all_findings.append(it)

    # PASS 1
    for idx, chunk in enumerate(chunks, start=1):
        resp = ollama_http.chat(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"NOTE (chunk {idx}/{len(chunks)}):\n{chunk}"},
            ],
            options=EXTRACTOR_OPTIONS,
            keep_alive=KEEP_ALIVE_EXTRACTOR,
        )
        raw = (resp.get("message", {}) or {}).get("content", "") or ""
        found = parse_findings_list_more_tolerant(raw)
        merge_findings(found)
        debug_parts.append(f"--- PASS1 chunk {idx}/{len(chunks)} ---\n{raw}\n")

    # PASS 2 (delta)
    for idx, chunk in enumerate(chunks, start=1):
        current_list = "\n".join([f"- {x}" for x in all_findings])
        delta_prompt = (
            "You previously extracted this list of findings:\n"
            f"{current_list}\n\n"
            "Now review ONLY the following note chunk and output ONLY NEW findings that are supported "
            "in this chunk but missing from the list above. If none, output nothing.\n\n"
            f"NOTE CHUNK:\n{chunk}"
        )

        resp = ollama_http.chat(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": delta_prompt},
            ],
            options=EXTRACTOR_OPTIONS,
            keep_alive=KEEP_ALIVE_EXTRACTOR,
        )
        raw = (resp.get("message", {}) or {}).get("content", "") or ""
        found = parse_findings_list_more_tolerant(raw)
        merge_findings(found)
        debug_parts.append(f"--- PASS2 delta chunk {idx}/{len(chunks)} ---\n{raw}\n")

    return "\n".join(debug_parts), all_findings

def parse_one_hpo_line(text: str) -> Tuple[str, str]:
    """
    Parse a single line like: [HP:0001234] Term Name
    Return (id, name) or ("","")
    """
    if not text:
        return "", ""
    text = strip_think_blocks(text)
    m = re.search(r"\[(HP:\d{7})\]\s*(.+)", text)
    if not m:
        m = re.search(r"^\s*(HP:\d{7})\s+(.+)$", text, flags=re.MULTILINE)
    if not m:
        m = re.search(r"^\s*(HP_\d{7})\s+(.+)$", text, flags=re.MULTILINE)
    if not m:
        return "", ""
    hid = normalize_hpo_id(m.group(1))
    name = remove_parentheticals(m.group(2).strip())
    return hid, name


# =====================
# Resume/checkpoint helpers
# =====================

def checkpoint_path(base_name: str) -> str:
    return f"{base_name}_OPT_A_checkpoint.json"

def load_checkpoint(base_name: str) -> Dict:
    path = checkpoint_path(base_name)
    if not os.path.exists(path):
        return {"completed_runs": 0, "term_counts": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"completed_runs": 0, "term_counts": {}}

def save_checkpoint(base_name: str, cp: Dict):
    path = checkpoint_path(base_name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2)
    os.replace(tmp, path)

def term_key(hpo_id: str, term_name: str) -> str:
    return f"{hpo_id}||{term_name}"


# =====================
# Qwen fallback (per-finding)
# =====================

def qwen_map_one_finding_with_tools(
    ollama_http: OllamaHTTP,
    mcp: SimpleMCPClient,
    finding: str,
) -> Tuple[str, str, str]:
    """
    Returns: (mapped_id, mapped_official_name, method)
      method: "qwen+mcp:...", "mcp_top1", "no_match"
    """
    tools = mcp.tools

    messages = [
        {"role": "system", "content": MAPPER_ONE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Finding: {finding}\n\nRemember: output ONLY one line [HP:########] Official Term Name, or NO_MATCH."},
    ]

    resp = ollama_http.chat(
        model=MAPPER_MODEL,
        messages=messages,
        tools=tools,
        options=MAPPER_OPTIONS,
        keep_alive=KEEP_ALIVE_MAPPER,
    )

    for _ in range(MAX_TOOL_ROUNDS):
        msg = resp.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break

        messages.append(msg)
        for tool in tool_calls:
            fn = tool.get("function", {}) or {}
            func_name = fn.get("name")
            args = fn.get("arguments", {})
            if not func_name:
                continue
            tool_result = mcp.call_tool(func_name, args)
            tool_msg = {"role": "tool", "name": func_name, "content": tool_result}
            if "id" in tool:
                tool_msg["tool_call_id"] = tool["id"]
            messages.append(tool_msg)

        resp = ollama_http.chat(
            model=MAPPER_MODEL,
            messages=messages,
            tools=tools,
            options=MAPPER_OPTIONS,
            keep_alive=KEEP_ALIVE_MAPPER,
        )

    final_text = (resp.get("message", {}) or {}).get("content", "") or ""
    if "NO_MATCH" in final_text.strip().upper():
        return "", "", "no_match"

    hid, name = parse_one_hpo_line(final_text)
    if hid and name:
        rid, rname, rvia = remap_against_hp_json(hid, name, FUZZY_MIN_SCORE)
        if rid and rname:
            return rid, rname, f"qwen+mcp:{rvia}"
        if normalize_hpo_id(hid) in VALID_HPO_IDS:
            rid = normalize_hpo_id(hid)
            return rid, ID_TO_OFFICIAL.get(rid, name), "qwen+mcp:kept_valid_id"

    if not USE_MCP_TOP1_IF_QWEN_FAILS:
        return "", "", "no_match"

    cands = mcp.search_hpo_terms(finding)
    if cands and isinstance(cands, list):
        top = cands[0]
        tid = normalize_hpo_id(str(top.get("id", "")))
        if tid in VALID_HPO_IDS:
            return tid, ID_TO_OFFICIAL.get(tid, top.get("name", "")), "mcp_top1"
    return "", "", "no_match"


# =====================
# Main processing
# =====================

def process_file_option_a(filepath: str, mcp: SimpleMCPClient, ollama_http: OllamaHTTP):
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    summary_csv = f"{base_name}_OPT_A_summary.csv"
    raw_csv = f"{base_name}_OPT_A_raw_log.csv"
    extractor_csv = f"{base_name}_OPT_A_extractor_log.csv"

    cp = load_checkpoint(base_name)
    already_done = int(cp.get("completed_runs", 0))

    print(f"\n=== Processing: {filepath} (Option A: DeepSeek chunked+delta -> local map -> Qwen/MCP fallback) ===")
    print(f"Resume: completed_runs={already_done} / {NUM_RUNS}")

    with open(filepath, "r", encoding="utf-8") as f:
        note_content = f.read()

    # rebuild counter from checkpoint
    term_counter = Counter()
    for k, v in (cp.get("term_counts", {}) or {}).items():
        term_counter[k] = v

    # ensure headers
    if not os.path.exists(raw_csv):
        with open(raw_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Run",
                "Finding",
                "Local Remap ID",
                "Local Remap Official Term",
                "Local Remap Method",
                "Fallback ID",
                "Fallback Official Term",
                "Fallback Method",
                "Final ID",
                "Final Official Term",
                "Final Method",
                "Is Valid ID in DB?",
            ])

    if not os.path.exists(extractor_csv):
        with open(extractor_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Run", "Extractor Debug (chunks+delta)", "Parsed Findings (joined)"])

    # run loop
    for run_i in range(already_done + 1, NUM_RUNS + 1):
        try:
            print(f"  > Run {run_i}/{NUM_RUNS}...", end="\r")

            # 1) DeepSeek high-recall extraction (chunked + delta)
            extractor_debug, findings = deepseek_extract_findings_high_recall(
                ollama_http,
                note_content,
                run_i,
                chunk_chars=EXTRACTOR_CHUNK_CHARS,
            )

            with open(extractor_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([run_i, extractor_debug, " | ".join(findings)])

            # 2) Local deterministic mapping
            mapped: List[Tuple[str, str, str, str]] = []  # (finding, id, official, method)
            unresolved: List[str] = []

            for fnd in findings:
                rid, rname, rvia = remap_against_hp_json("", fnd, FUZZY_MIN_SCORE)
                if rid and rname:
                    mapped.append((fnd, rid, rname, rvia))
                else:
                    unresolved.append(fnd)

            # 3) MCP fallback via Qwen, one finding at a time
            fallback_map: Dict[str, Tuple[str, str, str]] = {}  # finding -> (id, name, method)
            for fnd in unresolved:
                fid, fname, fvia = qwen_map_one_finding_with_tools(ollama_http, mcp, fnd)
                if fid and fname:
                    fallback_map[fnd] = (fid, fname, fvia)

            # 4) Choose final set for this run (dedupe by ID)
            run_unique_ids: Set[str] = set()
            run_terms: List[Tuple[str, str]] = []

            def add_run_term(hid: str, hname: str):
                if hid not in run_unique_ids:
                    run_unique_ids.add(hid)
                    run_terms.append((hid, hname))

            rows = []
            for fnd in findings:
                local = next((x for x in mapped if x[0] == fnd), None)
                local_id, local_name, local_via = ("", "", "")
                if local:
                    local_id, local_name, local_via = local[1], local[2], local[3]

                fb_id, fb_name, fb_via = ("", "", "")
                if fnd in fallback_map:
                    fb_id, fb_name, fb_via = fallback_map[fnd]

                # final decision: prefer local; else fallback
                final_id, final_name, final_via = ("", "", "")
                if local_id:
                    final_id, final_name, final_via = local_id, local_name, f"local:{local_via}"
                elif fb_id:
                    final_id, final_name, final_via = fb_id, fb_name, f"fallback:{fb_via}"
                else:
                    final_id, final_name, final_via = "", "", "unmapped"

                if final_id and final_name:
                    add_run_term(final_id, final_name)

                is_valid = str(final_id in VALID_HPO_IDS) if final_id else "False"

                rows.append([
                    run_i,
                    fnd,
                    local_id, local_name, local_via,
                    fb_id, fb_name, fb_via,
                    final_id, final_name, final_via,
                    is_valid,
                ])

            with open(raw_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(rows)

            # update summary counter once per run per unique ID
            for hid, hname in run_terms:
                term_counter[term_key(hid, hname)] += 1

            cp["term_counts"] = dict(term_counter)
            cp["completed_runs"] = run_i
            save_checkpoint(base_name, cp)

        except httpx.ReadTimeout:
            with open(raw_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([run_i, "ERROR", "", "", "", "", "", "", "", "", f"timeout>{OLLAMA_REQUEST_TIMEOUT_S}s", "False"])
            cp["completed_runs"] = run_i
            save_checkpoint(base_name, cp)
            print(f"\n  Error on run {run_i}: Ollama request timed out (> {OLLAMA_REQUEST_TIMEOUT_S}s)")
        except Exception as e:
            with open(raw_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([run_i, "ERROR", "", "", "", "", "", "", "", "", str(e), "False"])
            cp["completed_runs"] = run_i
            save_checkpoint(base_name, cp)
            print(f"\n  Error on run {run_i}: {e}")

    # write summary at end
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["HPO ID", "Official Term Name", f"Count (out of {NUM_RUNS})", "Percentage"])
        for k, count in term_counter.most_common():
            hid, hname = k.split("||", 1)
            w.writerow([hid, hname, count, f"{(count / NUM_RUNS) * 100:.1f}%"])

    print(f"\n  Saved summary to {summary_csv}")
    print(f"  Saved raw log to {raw_csv}")
    print(f"  Saved extractor log to {extractor_csv}")


def main():
    load_hpo_db(HPO_JSON_FILE)

    mcp = SimpleMCPClient(MCP_SERVER_URL)
    if not mcp.connect():
        return

    ollama_http = OllamaHTTP(OLLAMA_SERVER_URL, timeout_s=OLLAMA_REQUEST_TIMEOUT_S)

    try:
        mcp.initialize()
        mcp.list_tools()

        txt_files = glob.glob("*.txt")
        files_to_process = [f for f in txt_files if "summary" not in f and "log" not in f and "OPT_A" not in f]

        if not files_to_process:
            print("No .txt files found to process.")
            return

        for filepath in files_to_process:
            process_file_option_a(filepath, mcp, ollama_http)

    finally:
        try:
            mcp.close()
        except Exception:
            pass
        try:
            ollama_http.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
