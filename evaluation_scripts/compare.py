import os
import pandas as pd

# ----------------- CONFIG -----------------
FINAL_TERMS_FILE = "FINAL_TERMS.csv"
VALIDATED_FILE = "all_runs_validated.csv"

# Folder that contains your raw run logs
RAW_FOLDER = "raw_runs"

# Output files
OUTPUT_RUN_SUMMARY = "run_level_summary.csv"
OUTPUT_TERM_MATRIX = "term_recall_by_file.csv"
# ------------------------------------------


def load_final_terms(path):
    """
    Load FINAL_TERMS (tab-delimited) and build:
      - final_df: original dataframe
      - id_to_canonical: maps ANY accepted ID (or canonical ID) -> canonical ID
    Expected columns:
      HPO ID    Term Name    Accepted IDs
    """
    # The file is TAB-separated, not comma-separated
    final_df = pd.read_csv(path, sep="\t", dtype=str)

    # Normalize column names just in case
    final_df.columns = [c.strip() for c in final_df.columns]

    if "HPO ID" not in final_df.columns or "Term Name" not in final_df.columns:
        raise ValueError("FINAL_TERMS must have columns 'HPO ID' and 'Term Name'")

    if "Accepted IDs" not in final_df.columns:
        final_df["Accepted IDs"] = ""

    id_to_canonical = {}

    for _, row in final_df.iterrows():
        canonical_id = (row["HPO ID"] or "").strip()
        if not canonical_id:
            continue

        # map the canonical ID to itself
        id_to_canonical[canonical_id] = canonical_id

        accepted_raw = row["Accepted IDs"]
        if accepted_raw is None or str(accepted_raw).strip() == "":
            continue

        # strip quotes and whitespace, then split on commas
        cleaned = str(accepted_raw).strip().strip('"').strip("'")
        if not cleaned:
            continue

        for part in cleaned.split(","):
            alt_id = part.strip()
            if alt_id:
                id_to_canonical[alt_id] = canonical_id

    return final_df, id_to_canonical


def load_validation_truth(path):
    """
    Load all_runs_validated.csv and create a mapping:
       (HPO ID, Term Name) -> (Is_True_Term, Present)

    We treat each (ID, Name) pair separately.
    """
    vdf = pd.read_csv(path, dtype=str)

    vdf.columns = [c.strip() for c in vdf.columns]

    required = ["HPO ID", "Term Name", "Is_True_Term", "Present"]
    for col in required:
        if col not in vdf.columns:
            raise ValueError(f"all_runs_validated.csv must have column '{col}'")

    # Normalize ID and term name
    vdf["HPO ID"] = vdf["HPO ID"].astype(str).str.strip()
    vdf["Term Name"] = vdf["Term Name"].astype(str).str.strip()

    # Convert TRUE/FALSE strings to booleans
    for col in ["Is_True_Term", "Present"]:
        vdf[col] = (
            vdf[col]
            .astype(str)
            .str.upper()
            .map({"TRUE": True, "FALSE": False})
        )

    truth_dict = {}

    for _, row in vdf.iterrows():
        key = (row["HPO ID"], row["Term Name"])
        truth_dict[key] = (bool(row["Is_True_Term"]), bool(row["Present"]))

    return truth_dict


def process_raw_file(path, file_label, id_to_canonical, truth_dict):
    """
    Read a single raw run file and return:
      - df: dataframe with extra columns:
            Is_True_Term, Present, Is_Hallucination, Canonical_ID,
            Is_Target_Term, Is_Correct_Term, File
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Basic column check
    required = ["Run Number", "HPO ID", "Term Name"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Raw file '{path}' must have column '{col}'. "
                f"Columns found: {list(df.columns)}"
            )

    # Normalize ID and name to match validation file
    df["HPO ID"] = df["HPO ID"].astype(str).str.strip()
    df["Term Name"] = df["Term Name"].astype(str).str.strip()

    # Look up (HPO ID, Term Name) in validation truth
    is_true_list = []
    present_list = []

    for hpo_id, term_name in zip(df["HPO ID"], df["Term Name"]):
        key = (hpo_id, term_name)
        if key in truth_dict:
            is_true, present = truth_dict[key]
        else:
            # IMPORTANT:
            # If this exact ID+Name pair never appears in all_runs_validated,
            # we do NOT count it as hallucination by default.
            is_true, present = True, True

        is_true_list.append(is_true)
        present_list.append(present)

    df["Is_True_Term"] = is_true_list
    df["Present"] = present_list

    # Hallucination definition:
    # Only when this exact ID+Name pair is NOT true or NOT present
    df["Is_Hallucination"] = (~df["Is_True_Term"]) | (~df["Present"])

    # -------- TERM DETECTION / RECALL (HPO ID–based only) --------
    # Map to canonical IDs from FINAL_TERMS (including Accepted IDs)
    df["Canonical_ID"] = df["HPO ID"].map(id_to_canonical)

    # Term is "target" if its HPO ID (or accepted ID) appears in FINAL_TERMS
    df["Is_Target_Term"] = df["Canonical_ID"].notna()

    # Term is "correct" if it is a target term AND not hallucinated
    df["Is_Correct_Term"] = df["Is_Target_Term"] & (~df["Is_Hallucination"])

    # Keep file label for later summaries
    df["File"] = file_label

    return df


def build_run_level_summary(all_dfs):
    """
    all_dfs: list of per-file dataframes from process_raw_file()

    Returns a dataframe with columns:
      File, Run_Number, N_Hallucinations, N_Correct_Terms
    """
    summary_rows = []

    for df in all_dfs:
        file_label = df["File"].iloc[0]
        for run_number, sub in df.groupby("Run Number"):
            n_hall = int(sub["Is_Hallucination"].sum())
            n_correct = int(sub["Is_Correct_Term"].sum())
            summary_rows.append(
                {
                    "File": file_label,
                    "Run_Number": run_number,
                    "N_Hallucinations": n_hall,
                    "N_Correct_Terms": n_correct,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def build_term_recall_matrix(final_df, all_dfs):
    """
    Build a table where:
      - Rows: FINAL_TERMS (canonical HPO ID + Term Name)
      - Columns: each file label
      - Cell: % of runs in that file where this term was correctly picked
              (at least once in that run, and not hallucinated).
    """
    # canonical term list
    canonical_ids = list(final_df["HPO ID"].astype(str))
    term_names = dict(zip(final_df["HPO ID"], final_df["Term Name"]))

    # unique file labels
    file_labels = sorted({df["File"].iloc[0] for df in all_dfs})

    # init matrix
    matrix = pd.DataFrame(index=canonical_ids, columns=file_labels, dtype=float)

    # fill
    for df in all_dfs:
        file_label = df["File"].iloc[0]
        total_runs = df["Run Number"].nunique()

        if total_runs == 0:
            # Avoid division by zero
            matrix[file_label] = 0.0
            continue

        # Only consider non-hallucinated picks that map to a canonical ID
        df_correct = df[(~df["Is_Hallucination"]) & df["Canonical_ID"].notna()]

        # For each canonical term & run, check if present at least once
        if df_correct.empty:
            # No correct picks for any term in this file
            matrix[file_label] = 0.0
            continue

        per_term_run = (
            df_correct.groupby(["Canonical_ID", "Run Number"])
            .size()
            .reset_index(name="count")
        )

        # For each canonical ID, how many runs had it?
        counts = (
            per_term_run.groupby("Canonical_ID")["Run Number"]
            .nunique()
            .to_dict()
        )

        for cid in canonical_ids:
            n_runs_with_term = counts.get(cid, 0)
            perc = 100.0 * n_runs_with_term / total_runs
            matrix.loc[cid, file_label] = perc

    # Add term name as first column for readability
    matrix.insert(
        0,
        "Term Name",
        [term_names.get(cid, "") for cid in matrix.index],
    )

    # Make index explicit for the CSV
    matrix.insert(0, "HPO ID", matrix.index)

    # Reset index (so HPO ID becomes a normal column)
    matrix = matrix.reset_index(drop=True)

    return matrix


def main():
    # 1) Load gold standard and validation truth
    print("Loading FINAL_TERMS...")
    final_df, id_to_canonical = load_final_terms(FINAL_TERMS_FILE)

    print("Loading all_runs_validated truth (by HPO ID + Term Name)...")
    truth_dict = load_validation_truth(VALIDATED_FILE)

    # 2) Process all raw files
    if not os.path.isdir(RAW_FOLDER):
        raise SystemExit(
            f"Raw folder '{RAW_FOLDER}' does not exist. "
            f"Create it and place your raw run CSVs inside."
        )

    all_dfs = []
    raw_files = [
        f for f in os.listdir(RAW_FOLDER)
        if f.lower().endswith(".csv")
    ]

    if not raw_files:
        raise SystemExit(
            f"No CSV files found in '{RAW_FOLDER}'. "
            f"Put your raw run logs there."
        )

    for fname in sorted(raw_files):
        path = os.path.join(RAW_FOLDER, fname)
        file_label = os.path.splitext(fname)[0]  # e.g. 'grok_no_tools'
        print(f"Processing raw file: {path} ...")

        df = process_raw_file(
            path,
            file_label=file_label,
            id_to_canonical=id_to_canonical,
            truth_dict=truth_dict,
        )
        all_dfs.append(df)

    # 3) Build & write run-level summary
    print("Building run-level summary...")
    run_summary_df = build_run_level_summary(all_dfs)
    run_summary_df.to_csv(OUTPUT_RUN_SUMMARY, index=False)
    print(f"Run-level summary written to: {OUTPUT_RUN_SUMMARY}")

    # 4) Build & write term recall matrix
    print("Building term recall matrix...")
    term_matrix_df = build_term_recall_matrix(final_df, all_dfs)
    term_matrix_df.to_csv(OUTPUT_TERM_MATRIX, index=False)
    print(f"Term recall matrix written to: {OUTPUT_TERM_MATRIX}")


if __name__ == "__main__":
    main()
