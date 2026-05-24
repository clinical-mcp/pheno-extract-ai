import pandas as pd
import numpy as np
import os
import re
import json
import warnings
from collections import defaultdict
from scipy.stats import ttest_ind

# ================= CONFIGURATION =================
RAW_RUNS_FOLDER = "raw_runs"
FINAL_TERMS_PATH = "final_terms.csv"
REVIEWED_FILE_PATH = "terms_annotated.csv"
OUTPUT_FOLDER = "analysis_results"
HPO_JSON_PATH = "hp.json"
# --- Optional debug export (set to True to emit per-term diagnostics for a single run) ---
DEBUG_DIAGNOSTIC = True
DEBUG_MODEL = "CLAUDE"
DEBUG_MODE = "No Tools"
DEBUG_RUN_NUMBER = 1
# =================================================


def pick_col_exact(df: pd.DataFrame, candidates):
    cand_lower = [c.strip().lower() for c in candidates]
    for c in df.columns:
        if str(c).strip().lower() in cand_lower:
            return c
    return None


def find_support_col(df: pd.DataFrame):
    exact = pick_col_exact(
        df,
        [
            "Support",
            "MANUAL_Present_In_Note_Support",
            "Present_In_Note_Support",
            "Inferred/Direct",
            "Support_Type",
        ],
    )
    if exact:
        return exact

    for c in df.columns:
        name = str(c).strip().lower()
        if any(k in name for k in ["support", "direct", "inferred"]):
            vals = (
                df[c]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .head(200)
                .tolist()
            )
            if any(v in ("direct", "inferred") for v in vals):
                return c
    return None


def norm_id(val: str) -> str:
    v = str(val).strip()
    if not v or v.lower() == "nan":
        return ""
    if v.startswith("HP:"):
        return v
    if re.match(r"^\d+$", v):
        return "HP:" + v
    return v


def norm_support(val: str) -> str:
    s = str(val).strip().lower()
    if s in ("direct", "inferred"):
        return s
    return ""


HP_IRI_PREFIX = "http://purl.obolibrary.org/obo/HP_"


def to_hp_id(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if raw.startswith(HP_IRI_PREFIX):
        return "HP:" + raw[len(HP_IRI_PREFIX) :]
    if raw.startswith("HP:"):
        return raw
    if re.match(r"^\d{7}$", raw):
        return "HP:" + raw
    return ""


def clean_term_for_matching(term: str) -> str:
    if not isinstance(term, str):
        return ""
    cleaned = re.sub(r"\s*\(.*?\)", "", term)
    return cleaned.strip()


def normalize_term_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def safe_int(val, default=0):
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def load_hpo_terms(hpo_json_path: str):
    with open(hpo_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_name = {}
    id_to_synonyms = defaultdict(list)
    term_lookup = set()

    if isinstance(data, dict) and "graphs" in data:
        for graph in data["graphs"]:
            nodes = graph.get("nodes", [])
            for node in nodes:
                term_id = to_hp_id(node.get("id", ""))
                if not term_id:
                    continue

                label = node.get("lbl") or node.get("label") or node.get("name")
                if isinstance(label, str) and label.strip():
                    id_to_name[term_id] = label
                    cleaned = clean_term_for_matching(label)
                    norm_label = normalize_term_name(cleaned)
                    if norm_label:
                        term_lookup.add(norm_label)

                meta = node.get("meta", {})
                syn_list = meta.get("synonyms", [])
                for syn in syn_list:
                    val = syn.get("val")
                    if isinstance(val, str) and val.strip():
                        id_to_synonyms[term_id].append(val)
                        cleaned = clean_term_for_matching(val)
                        norm_syn = normalize_term_name(cleaned)
                        if norm_syn:
                            term_lookup.add(norm_syn)

    return id_to_name, id_to_synonyms, term_lookup


def load_ground_truth_and_id_map(path: str):
    df = pd.read_csv(path).copy()

    normalized_id_col = pick_col_exact(df, ["Normalized_ID"])
    canonical_id_col = pick_col_exact(df, ["Canonical_ID", "HPO ID", "HPO_ID", "HPOID"])
    canonical_name_col = pick_col_exact(df, ["Canonical_Official_Term_Name", "Term Name", "Term_Name", "Name"])
    accepted_ids_col = pick_col_exact(df, ["Accepted IDs", "Accepted_IDs", "Accepted"])
    support_col = find_support_col(df)

    # ---------- mapping-style ----------
    if normalized_id_col and canonical_id_col and canonical_name_col:
        df[normalized_id_col] = df[normalized_id_col].apply(norm_id)
        df[canonical_id_col] = df[canonical_id_col].apply(norm_id)
        df[canonical_name_col] = df[canonical_name_col].astype(str).str.strip()
        if support_col:
            df[support_col] = df[support_col].apply(norm_support)
        else:
            df["__support"] = ""
            support_col = "__support"

        by_primary = {}
        for _, r in df.iterrows():
            primary = str(r[canonical_id_col]).strip()
            if not primary:
                continue

            name = str(r[canonical_name_col]).strip()
            if not name or name.lower() == "nan":
                name = primary

            nid = str(r[normalized_id_col]).strip()
            sup = norm_support(r[support_col])

            if primary not in by_primary:
                by_primary[primary] = {
                    "name": name,
                    "primary_id": primary,
                    "all_ids": set([primary]),
                    "support": "",
                }

            if nid:
                by_primary[primary]["all_ids"].add(nid)

            if sup == "direct":
                by_primary[primary]["support"] = "direct"
            elif sup == "inferred" and by_primary[primary]["support"] != "direct":
                by_primary[primary]["support"] = "inferred"

            if by_primary[primary]["name"] == primary and name != primary:
                by_primary[primary]["name"] = name

        ground_truth = sorted(by_primary.values(), key=lambda x: (x["name"], x["primary_id"]))

        id_to_primary = {}
        for gt in ground_truth:
            for a in gt["all_ids"]:
                id_to_primary[a] = gt["primary_id"]

        return ground_truth, id_to_primary

    # ---------- legacy-style ----------
    hpo_id_col = pick_col_exact(df, ["HPO ID", "HPO_ID", "HPOID", "Canonical_ID"])
    term_name_col = pick_col_exact(df, ["Term Name", "Term_Name", "Name", "Canonical_Official_Term_Name"])

    if not (hpo_id_col and term_name_col):
        raise ValueError(
            "final_terms.csv not recognized. Provide either:\n"
            "  mapping-style: Normalized_ID, Canonical_ID/HPO ID, Canonical_Official_Term_Name/Term Name\n"
            "or legacy-style: HPO ID, Term Name, (optional) Accepted IDs"
        )

    df[hpo_id_col] = df[hpo_id_col].apply(norm_id)
    if support_col:
        df[support_col] = df[support_col].apply(norm_support)

    by_primary = {}
    for _, r in df.iterrows():
        primary = str(r[hpo_id_col]).strip()
        if not primary:
            continue

        name = str(r[term_name_col]).strip()
        if not name or name.lower() == "nan":
            name = primary

        all_ids = set([primary])
        if accepted_ids_col and pd.notna(r.get(accepted_ids_col, np.nan)):
            for alt in str(r[accepted_ids_col]).split(";"):
                a = norm_id(alt)
                if a:
                    all_ids.add(a)

        sup = norm_support(r.get(support_col, "")) if support_col else ""

        if primary not in by_primary:
            by_primary[primary] = {
                "name": name,
                "primary_id": primary,
                "all_ids": set(),
                "support": "",
            }

        by_primary[primary]["all_ids"].update(all_ids)
        if sup == "direct":
            by_primary[primary]["support"] = "direct"
        elif sup == "inferred" and by_primary[primary]["support"] != "direct":
            by_primary[primary]["support"] = "inferred"

    ground_truth = sorted(by_primary.values(), key=lambda x: (x["name"], x["primary_id"]))
    id_to_primary = {}
    for gt in ground_truth:
        for a in gt["all_ids"]:
            id_to_primary[a] = gt["primary_id"]

    return ground_truth, id_to_primary


def parse_model_info(filename: str):
    """
    Parse model name and tools mode from standardized filename.
    
    Expected format: sim{N}_nw_{MODEL}_{WITH_TOOLS|NO_TOOLS}_{type}.csv
    Examples:
        sim1_nw_CLAUDE_WITH_TOOLS_raw_log.csv -> ("CLAUDE", "With Tools")
        sim1_nw_GPT_NO_TOOLS_raw_log.csv -> ("GPT", "No Tools")
    """
    base = os.path.basename(filename)
    
    # Primary pattern: sim{N}_nw_{MODEL}_{WITH_TOOLS|NO_TOOLS}_{type}.csv
    m = re.match(r"sim\d+_nw_([A-Za-z0-9]+)_(WITH_TOOLS|NO_TOOLS|WITH_MCP)_(?:raw_log|cost_log)\.csv$", base, re.IGNORECASE)
    if m:
        model_name = m.group(1).upper()
        mode_raw = m.group(2).upper()
        if mode_raw == "NO_TOOLS":
            return model_name, "No Tools"
        elif mode_raw in ("WITH_TOOLS", "WITH_MCP"):
            return model_name, "With Tools"
    
    # Fallback: try to extract from older pattern (captures everything before _raw_log)
    m = re.search(r"sim\d+_nw_(.+?)_raw_log(?:\.csv)?$", base, re.IGNORECASE)
    if m:
        cfg = m.group(1)
        if cfg.endswith("_NO_TOOLS"):
            return cfg[: -len("_NO_TOOLS")].upper(), "No Tools"
        if cfg.endswith("_WITH_TOOLS"):
            return cfg[: -len("_WITH_TOOLS")].upper(), "With Tools"
        if cfg.endswith("_WITH_MCP"):
            return cfg[: -len("_WITH_MCP")].upper(), "With Tools"
        # Handle partial matches like "CLAUDE_WITH" (missing TOOLS)
        if cfg.endswith("_WITH"):
            return cfg[: -len("_WITH")].upper(), "With Tools"
        if cfg.upper() == "EPIC":
            return "EPIC", "Comparison"
        return cfg.upper(), "Unknown"
    
    return "Unknown", "Unknown"


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading data...")
    ground_truth, id_to_primary = load_ground_truth_and_id_map(FINAL_TERMS_PATH)
    hpo_id_to_name, hpo_id_to_synonyms, hpo_term_lookup = load_hpo_terms(HPO_JSON_PATH)
    review_df = pd.read_csv(REVIEWED_FILE_PATH)

    gt_term_to_ids = defaultdict(set)
    gt_term_lookup = {
        normalize_term_name(clean_term_for_matching(gt["name"])) for gt in ground_truth
    }
    gt_term_lookup = {t for t in gt_term_lookup if t}
    gt_synonym_lookup = set()
    for gt in ground_truth:
        hpo_id = gt["primary_id"]
        name_norm = normalize_term_name(clean_term_for_matching(gt["name"]))
        if name_norm:
            gt_term_to_ids[name_norm].add(hpo_id)
        label = hpo_id_to_name.get(hpo_id, "")
        if label:
            norm_label = normalize_term_name(clean_term_for_matching(label))
            if norm_label:
                gt_synonym_lookup.add(norm_label)
                gt_term_to_ids[norm_label].add(hpo_id)
        for syn in hpo_id_to_synonyms.get(hpo_id, []):
            norm_syn = normalize_term_name(clean_term_for_matching(syn))
            if norm_syn:
                gt_synonym_lookup.add(norm_syn)
                gt_term_to_ids[norm_syn].add(hpo_id)

    gt_term_lookup = gt_term_lookup.union(gt_synonym_lookup)

    gt_ids_all = set(gt["primary_id"] for gt in ground_truth)
    gt_ids_direct = set(gt["primary_id"] for gt in ground_truth if gt.get("support") == "direct")
    gt_ids_inferred = set(gt["primary_id"] for gt in ground_truth if gt.get("support") == "inferred")
    print(
        f"Ground truth unique terms: {len(gt_ids_all)} | direct: {len(gt_ids_direct)} | inferred: {len(gt_ids_inferred)}"
    )

    pd.DataFrame(
        [
            {
                "HPO ID": gt["primary_id"],
                "Term Name": gt["name"],
                "Support": gt.get("support", ""),
                "Accepted IDs": ";".join(sorted(gt["all_ids"])),
            }
            for gt in ground_truth
        ]
    ).to_csv(os.path.join(OUTPUT_FOLDER, "GroundTruth_Unique_Canonical_Terms.csv"), index=False)

    # error lookup
    nid_col = pick_col_exact(review_df, ["Normalized_ID"])
    raw_term_col = pick_col_exact(review_df, ["Raw_Term"])
    err_col = pick_col_exact(review_df, ["MANUAL_Error_Type"])
    cleaned_term_col = pick_col_exact(review_df, ["Cleaned_Term_Used"])
    if not (nid_col and raw_term_col and err_col):
        raise SystemExit("terms_annotated.csv must contain: Normalized_ID, Raw_Term, MANUAL_Error_Type")

    error_lookup = {}
    cleaned_term_lookup = {}
    for _, r in review_df.iterrows():
        key = (norm_id(r[nid_col]), str(r[raw_term_col]).strip())
        try:
            code = int(r[err_col])
        except Exception:
            code = 0
        error_lookup[key] = code
        if cleaned_term_col:
            cleaned_val = str(r[cleaned_term_col]).strip()
            if cleaned_val and cleaned_val.lower() != "nan":
                cleaned_term_lookup[key] = cleaned_val

    print(f"Processing runs from {RAW_RUNS_FOLDER}...")

    all_runs_data = []
    hallucination_rows = []
    term_stats = {gt["primary_id"]: {} for gt in ground_truth}
    model_run_counts = {}
    diag_rows = []
    diag_canonical_rows = []

    # Only process raw_log files (not cost_log files)
    files = [f for f in os.listdir(RAW_RUNS_FOLDER) if f.endswith("_raw_log.csv")]
    
    # Also find cost_log files for cost analysis
    cost_files = [f for f in os.listdir(RAW_RUNS_FOLDER) if f.endswith("_cost_log.csv")]
    
    print(f"Found {len(files)} raw_log files to process:")
    for f in sorted(files):
        model, mode = parse_model_info(f)
        print(f"  {f} -> Model: {model}, Mode: {mode}")
    print()
    
    print(f"Found {len(cost_files)} cost_log files:")
    for f in sorted(cost_files):
        print(f"  {f}")
    print()

    for fname in files:
        model_name, mode = parse_model_info(fname)
        config_name = f"{model_name} ({mode})"

        if config_name not in model_run_counts:
            model_run_counts[config_name] = 0
            for t_id in term_stats:
                term_stats[t_id][config_name] = 0

        df = pd.read_csv(os.path.join(RAW_RUNS_FOLDER, fname))

        def get_col(candidates):
            for c in df.columns:
                if str(c).strip().lower() in [x.lower() for x in candidates]:
                    return c
            return None

        run_col = get_col(["Run Number", "Run", "Run_Number"])
        id_col = get_col(["HPO ID", "ID", "HPOID", "Model HPO ID", "Model_HPO_ID"])
        term_col = get_col(["Term Name", "Term", "Label", "Name", "Model Term Name", "Model_Term_Name"])
        if not (run_col and id_col and term_col):
            print(f"Skipping {fname}: Missing columns")
            continue

        for run_id, group in df.groupby(run_col):
            has_valid_hpo = False
            for _, row in group.iterrows():
                val = str(row[id_col]).strip()
                if val.startswith("HP:") or (val.isdigit() and len(val) >= 4):
                    has_valid_hpo = True
                    break
            if not has_valid_hpo:
                continue

            model_run_counts[config_name] += 1

            canonical_out = {}  # canonical_id -> set(error_codes)
            term_present_by_canonical = {}
            term_present_in_gt_by_canonical = {}
            run_correct_ids = set()
            run_term_only_ids = set()

            run_is_debug = (
                DEBUG_DIAGNOSTIC
                and model_name == DEBUG_MODEL
                and mode == DEBUG_MODE
                and safe_int(run_id, -1) == safe_int(DEBUG_RUN_NUMBER, -2)
            )

            for _, row in group.iterrows():
                raw_id = norm_id(row[id_col])
                raw_term = str(row[term_col]).strip()
                if not raw_id:
                    continue

                canonical_id = id_to_primary.get(raw_id, raw_id)

                cleaned_term = cleaned_term_lookup.get(
                    (raw_id, raw_term), clean_term_for_matching(raw_term)
                )
                norm_cleaned = normalize_term_name(cleaned_term)
                is_real_term = False
                is_present_in_gt = False
                matched_gt_ids = set()
                if norm_cleaned:
                    is_real_term = norm_cleaned in hpo_term_lookup
                    if is_real_term:
                        matched_gt_ids = gt_term_to_ids.get(norm_cleaned, set())
                        is_present_in_gt = bool(matched_gt_ids) or (canonical_id in gt_ids_all)

                code = error_lookup.get((raw_id, raw_term), None)
                if code is None and canonical_id != raw_id:
                    code = error_lookup.get((canonical_id, raw_term), 0)
                if code is None:
                    code = 0

                canonical_out.setdefault(canonical_id, set()).add(int(code))
                term_present_by_canonical[canonical_id] = term_present_by_canonical.get(
                    canonical_id, False
                ) or is_real_term
                term_present_in_gt_by_canonical[canonical_id] = term_present_in_gt_by_canonical.get(
                    canonical_id, False
                ) or is_present_in_gt

                if int(code) == 0 and canonical_id in gt_ids_all:
                    run_correct_ids.add(canonical_id)
                    run_term_only_ids.add(canonical_id)
                elif int(code) == 1 and is_real_term and matched_gt_ids:
                    run_term_only_ids.update(matched_gt_ids)

                if run_is_debug:
                    diag_rows.append(
                        {
                            "Model": model_name,
                            "Mode": mode,
                            "Run_Number": run_id,
                            "Raw_ID": raw_id,
                            "Raw_Term": raw_term,
                            "Canonical_ID": canonical_id,
                            "Error_Code": int(code),
                            "Cleaned_Term_Used": cleaned_term,
                            "Normalized_Cleaned_Term": norm_cleaned,
                            "Term_In_HPO_Lookup": is_real_term,
                            "Term_In_GT_Lookup": is_present_in_gt,
                            "Canonical_In_GT": canonical_id in gt_ids_all,
                        }
                    )

                if int(code) in (2, 3):
                    hallucination_rows.append(
                        {
                            "Model": model_name,
                            "Mode": mode,
                            "Run_Number": run_id,
                            "Raw_ID": raw_id,
                            "Raw_Term": raw_term,
                            "Canonical_ID": canonical_id,
                            "Error_Code": int(code),
                        }
                    )

            run_produced_canonicals = set(canonical_out.keys())
            run_total_unique_terms = len(run_produced_canonicals)

            run_mapping_errors = sum(1 for codes in canonical_out.values() if (1 in codes) or (2 in codes))
            run_hallucinations = sum(1 for codes in canonical_out.values() if (2 in codes) or (3 in codes))

            correct_terms_count = len(run_correct_ids)
            correct_terms_count_term_only = len(run_term_only_ids)

            if run_is_debug:
                for cid in sorted(run_produced_canonicals):
                    diag_canonical_rows.append(
                        {
                            "Model": model_name,
                            "Mode": mode,
                            "Run_Number": run_id,
                            "Canonical_ID": cid,
                            "Canonical_In_GT": cid in gt_ids_all,
                            "Term_In_HPO_Lookup": term_present_by_canonical.get(cid, False),
                            "Term_In_GT_Lookup": term_present_in_gt_by_canonical.get(cid, False),
                            "Error_Codes": ";".join(sorted(str(x) for x in canonical_out.get(cid, []))),
                        }
                    )

            caught_direct = len(run_produced_canonicals.intersection(gt_ids_direct))
            caught_inferred = len(run_produced_canonicals.intersection(gt_ids_inferred))

            total_gt = len(gt_ids_all)
            total_direct = len(gt_ids_direct)
            total_inferred = len(gt_ids_inferred)

            acc_overall = correct_terms_count / total_gt if total_gt else 0.0
            acc_term_only = correct_terms_count_term_only / total_gt if total_gt else 0.0
            acc_direct = caught_direct / total_direct if total_direct else 0.0
            acc_inferred = caught_inferred / total_inferred if total_inferred else 0.0

            for gt_id in run_correct_ids:
                term_stats[gt_id][config_name] += 1

            all_runs_data.append(
                {
                    "Model": model_name,
                    "Mode": mode,
                    "Run_Number": run_id,
                    "Total_Output_Terms_Unique": run_total_unique_terms,
                    "Total_Output_Terms_Real_(Type0+Type3)": sum(
                        1 for codes in canonical_out.values() if (0 in codes) or (3 in codes)
                    ),
                    "Mapping_Error_Count_(Type1+Type2)": run_mapping_errors,
                    "Hallucination_Count_(Type2+Type3)": run_hallucinations,
                    "Correct_Terms_Caught_Unique": correct_terms_count,
                    "Correct_Terms_Caught_Term_Only": correct_terms_count_term_only,
                    "Accuracy_Recall_Overall": acc_overall,
                    "Accuracy_Recall_Term_Only": acc_term_only,
                    "Accuracy_Recall_Direct": acc_direct,
                    "Accuracy_Recall_Inferred": acc_inferred,
                    "Mapping_Error_Rate": (run_mapping_errors / run_total_unique_terms) if run_total_unique_terms > 0 else 0.0,
                    "Hallucination_Rate": (run_hallucinations / run_total_unique_terms) if run_total_unique_terms > 0 else 0.0,
                    "Raw_Term_Present_In_HPO_Rate": (
                        sum(1 for v in term_present_by_canonical.values() if v)
                        / run_total_unique_terms
                    )
                    if run_total_unique_terms > 0
                    else 0.0,
                }
            )

    results_df = pd.DataFrame(all_runs_data)

    if DEBUG_DIAGNOSTIC and diag_rows:
        pd.DataFrame(diag_rows).to_csv(
            os.path.join(OUTPUT_FOLDER, "Debug_Term_Only_Details.csv"), index=False
        )
    if DEBUG_DIAGNOSTIC and diag_canonical_rows:
        pd.DataFrame(diag_canonical_rows).to_csv(
            os.path.join(OUTPUT_FOLDER, "Debug_Term_Only_Canonicals.csv"), index=False
        )

    # ================= GOAL 1: Term Detection Rates =================
    print("Generating Term Detection Rates (unique)...")
    all_configs = sorted(model_run_counts.keys())

    term_rows = []
    for gt in ground_truth:
        support = gt.get("support", "")
        row = {
            "HPO ID": gt["primary_id"],
            "Term Name": gt["name"],
            "Support": support,
            "Direct": 1 if support == "direct" else 0,
            "Inferred": 1 if support == "inferred" else 0,
        }
        for config in all_configs:
            n = model_run_counts[config]
            caught = term_stats[gt["primary_id"]].get(config, 0)
            perc = (caught / n * 100) if n > 0 else 0.0
            row[f"{config} (N={n})"] = f"{perc:.2f}%"
        term_rows.append(row)

    pd.DataFrame(term_rows).to_csv(os.path.join(OUTPUT_FOLDER, "Term_Detection_Rates.csv"), index=False)

    # ================= GOAL 2: Run Stats =================
    print("Generating Run Stats...")
    results_df.sort_values(["Model", "Mode", "Run_Number"]).to_csv(
        os.path.join(OUTPUT_FOLDER, "Run_Performance_Stats.csv"), index=False
    )

    # ================= GOAL 2a: Model Performance Summary =================
    print("Generating Model Performance Summary...")
    if not results_df.empty:
        summary_df = (
            results_df.groupby(["Model", "Mode"], as_index=False)
            .agg(
                N_Runs=("Run_Number", "count"),
                Accuracy_Recall_Overall_Mean=("Accuracy_Recall_Overall", "mean"),
                Accuracy_Recall_Term_Only_Mean=("Accuracy_Recall_Term_Only", "mean"),
                Raw_Term_Present_In_HPO_Rate_Mean=("Raw_Term_Present_In_HPO_Rate", "mean"),
                Mapping_Error_Rate_Mean=("Mapping_Error_Rate", "mean"),
                Hallucination_Rate_Mean=("Hallucination_Rate", "mean"),
                Mapping_Error_Count_Mean=("Mapping_Error_Count_(Type1+Type2)", "mean"),
                Hallucination_Count_Mean=("Hallucination_Count_(Type2+Type3)", "mean"),
            )
            .sort_values(["Model", "Mode"])
        )
        summary_df.to_csv(
            os.path.join(OUTPUT_FOLDER, "Model_Performance_Summary.csv"), index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "Model",
                "Mode",
                "N_Runs",
                "Accuracy_Recall_Overall_Mean",
                "Accuracy_Recall_Term_Only_Mean",
                "Raw_Term_Present_In_HPO_Rate_Mean",
                "Mapping_Error_Rate_Mean",
                "Hallucination_Rate_Mean",
                "Mapping_Error_Count_Mean",
                "Hallucination_Count_Mean",
            ]
        ).to_csv(
            os.path.join(OUTPUT_FOLDER, "Model_Performance_Summary.csv"), index=False
        )

    # ================= GOAL 2b: Hallucination List =================
    print("Generating Hallucination List...")
    hallucinations_df = pd.DataFrame(hallucination_rows)
    if not hallucinations_df.empty:
        hallucinations_df.sort_values(["Model", "Mode", "Run_Number"]).to_csv(
            os.path.join(OUTPUT_FOLDER, "Hallucinations_By_Model_Mode.csv"), index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "Model",
                "Mode",
                "Run_Number",
                "Raw_ID",
                "Raw_Term",
                "Canonical_ID",
                "Error_Code",
            ]
        ).to_csv(
            os.path.join(OUTPUT_FOLDER, "Hallucinations_By_Model_Mode.csv"), index=False
        )

    # ================= GOAL 3: Statistical Comparison -> CSV table (ALWAYS compute p-values) =================
    print("Performing Statistical Analysis...")
    stats_rows = []

    if not results_df.empty:
        required_modes = {"With Tools", "No Tools"}
        for model in results_df["Model"].unique():
            model_data = results_df[results_df["Model"] == model]
            available_modes = set(model_data["Mode"].unique())
            if not required_modes.issubset(available_modes):
                continue

            tools_data = model_data[model_data["Mode"] == "With Tools"]
            no_tools_data = model_data[model_data["Mode"] == "No Tools"]

            metrics = [
                ("Accuracy_Recall_Overall", "Accuracy (Recall) - Overall"),
                ("Accuracy_Recall_Term_Only", "Accuracy (Recall) - Term Only"),
                ("Accuracy_Recall_Direct", "Accuracy (Recall) - Direct"),
                ("Accuracy_Recall_Inferred", "Accuracy (Recall) - Inferred"),
                ("Raw_Term_Present_In_HPO_Rate", "Raw Term Present in HPO Rate"),
                ("Mapping_Error_Rate", "Mapping Error Rate (Type 1+2)"),
                ("Hallucination_Rate", "Hallucination Rate (Type 2+3)"),
            ]

            for col, label in metrics:
                g1 = tools_data[col]
                g2 = no_tools_data[col]

                n1 = int(g1.dropna().shape[0])
                n2 = int(g2.dropna().shape[0])
                mean_tools = g1.mean()
                mean_no_tools = g2.mean()

                p_val = np.nan
                note = ""

                if n1 == 0 or n2 == 0:
                    note = "One group has no data after NaN removal"
                else:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        try:
                            _, p_val = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
                        except Exception as e:
                            note = f"ttest_ind error: {e}"
                            p_val = np.nan

                        if w:
                            msgs = "; ".join(str(x.message) for x in w)
                            note = (note + " | " if note else "") + msgs

                sig = ""
                if pd.notna(p_val):
                    sig = "YES" if p_val < 0.05 else "NO"

                stats_rows.append(
                    {
                        "Model": model,
                        "Metric": label,
                        "With Tools N": n1,
                        "No Tools N": n2,
                        "With Tools Mean": mean_tools,
                        "No Tools Mean": mean_no_tools,
                        "P-Value": p_val,
                        "Significant(p<0.05)": sig,
                        "Note": note,
                    }
                )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUTPUT_FOLDER, "Statistical_Significance.csv"), index=False)

    # ================= GOAL 4: Cost Analysis =================
    print("Analyzing Cost Data...")
    
    # Build a mapping from (model, mode) -> cost_log filename
    cost_file_map = {}
    for cf in cost_files:
        model, mode = parse_model_info(cf)
        config_key = (model, mode)
        cost_file_map[config_key] = cf
    
    # Collect cost data for each configuration
    cost_data = {}  # config_name -> list of estimated costs per run
    
    for cf in cost_files:
        model, mode = parse_model_info(cf)
        config_name = f"{model} ({mode})"
        
        try:
            cost_df = pd.read_csv(os.path.join(RAW_RUNS_FOLDER, cf))
            
            # Find the estimated cost column (flexible matching)
            cost_col = None
            for c in cost_df.columns:
                col_lower = str(c).strip().lower()
                if "estimated" in col_lower or "cost" in col_lower:
                    cost_col = c
                    break
            
            if cost_col is None:
                print(f"  Warning: No cost column found in {cf}")
                continue
            
            # Get cost per run
            run_col = None
            for c in cost_df.columns:
                col_lower = str(c).strip().lower()
                if "run" in col_lower:
                    run_col = c
                    break
            
            if run_col:
                # Sum cost by run if multiple rows per run, or just take the value
                run_costs = cost_df.groupby(run_col)[cost_col].sum().tolist()
            else:
                # Assume each row is a run
                run_costs = cost_df[cost_col].tolist()
            
            if config_name not in cost_data:
                cost_data[config_name] = []
            cost_data[config_name].extend(run_costs)
            
            print(f"  {cf} -> {len(run_costs)} runs, costs: {run_costs}")
            
        except Exception as e:
            print(f"  Error reading {cf}: {e}")
    
    # Build cost summary table
    cost_summary_rows = []
    
    # Get all unique models and modes from both raw files and cost files
    all_models = set()
    all_modes = set(["No Tools", "With Tools"])
    
    for config_name in list(model_run_counts.keys()) + list(cost_data.keys()):
        # Parse model from config_name like "CLAUDE (With Tools)"
        m = re.match(r"(.+?) \((No Tools|With Tools|Unknown)\)", config_name)
        if m:
            all_models.add(m.group(1))
    
    for model in sorted(all_models):
        for mode in ["No Tools", "With Tools"]:
            config_name = f"{model} ({mode})"
            
            costs = cost_data.get(config_name, [])
            n_runs = model_run_counts.get(config_name, 0)
            
            if costs:
                avg_cost = np.mean(costs)
                total_cost = np.sum(costs)
                min_cost = np.min(costs)
                max_cost = np.max(costs)
                n_cost_runs = len(costs)
            else:
                avg_cost = np.nan
                total_cost = np.nan
                min_cost = np.nan
                max_cost = np.nan
                n_cost_runs = 0
            
            cost_summary_rows.append({
                "Model": model,
                "Mode": mode,
                "N_Runs_Performance": n_runs,
                "N_Runs_Cost": n_cost_runs,
                "Avg_Cost_Per_Run": avg_cost if n_cost_runs > 0 else "N/A",
                "Total_Cost": total_cost if n_cost_runs > 0 else "N/A",
                "Min_Cost": min_cost if n_cost_runs > 0 else "N/A",
                "Max_Cost": max_cost if n_cost_runs > 0 else "N/A",
            })
    
    cost_summary_df = pd.DataFrame(cost_summary_rows)
    cost_summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "Cost_Analysis.csv"), index=False)
    
    # Also add cost comparison between Tools vs No Tools for each model
    cost_comparison_rows = []
    for model in sorted(all_models):
        tools_costs = cost_data.get(f"{model} (With Tools)", [])
        no_tools_costs = cost_data.get(f"{model} (No Tools)", [])
        
        tools_avg = np.mean(tools_costs) if tools_costs else np.nan
        no_tools_avg = np.mean(no_tools_costs) if no_tools_costs else np.nan
        
        # Calculate cost difference
        if tools_costs and no_tools_costs:
            cost_diff = tools_avg - no_tools_avg
            cost_ratio = tools_avg / no_tools_avg if no_tools_avg > 0 else np.nan
        else:
            cost_diff = np.nan
            cost_ratio = np.nan
        
        cost_comparison_rows.append({
            "Model": model,
            "With_Tools_N": len(tools_costs),
            "With_Tools_Avg_Cost": tools_avg if tools_costs else "N/A",
            "No_Tools_N": len(no_tools_costs),
            "No_Tools_Avg_Cost": no_tools_avg if no_tools_costs else "N/A",
            "Cost_Difference": cost_diff if not np.isnan(cost_diff) else "N/A",
            "Cost_Ratio_(Tools/NoTools)": cost_ratio if not np.isnan(cost_ratio) else "N/A",
        })
    
    cost_comparison_df = pd.DataFrame(cost_comparison_rows)
    cost_comparison_df.to_csv(os.path.join(OUTPUT_FOLDER, "Cost_Comparison.csv"), index=False)
    
    print("\nCost Summary:")
    print(cost_summary_df.to_string(index=False))

    print(f"\nAnalysis Complete. Files saved to: {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
