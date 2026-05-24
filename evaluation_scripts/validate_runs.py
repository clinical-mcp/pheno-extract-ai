import json
import os
import re
from collections import defaultdict
import pandas as pd

# ---------- CONFIG ----------
HPO_JSON_PATH = "hp.json"
INPUT_FOLDER = "."                  # use current folder
OUTPUT_FOLDER = "validated_csvs"    # write outputs here
# ----------------------------

HP_IRI_PREFIX = "http://purl.obolibrary.org/obo/HP_"

def to_hp_id(raw):
    """Normalize raw ID to 'HP:XXXXXXX'."""
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if raw.startswith(HP_IRI_PREFIX):
        local = raw[len(HP_IRI_PREFIX):]
        return "HP:" + local
    if raw.startswith("HP:"):
        return raw
    if re.match(r'^\d{7}$', raw):
        return "HP:" + raw
    return None

def clean_term_for_matching(term):
    """
    Remove parenthetical explanations to improve matching rates.
    Example: "Small for gestational age (birth weight 4lbs)" -> "Small for gestational age"
    """
    if not isinstance(term, str):
        return ""
    # Regex: Remove space + opening paren + any content + closing paren
    # non-greedy match (.*?) handles cases with multiple parens gracefully
    cleaned = re.sub(r'\s*\(.*?\)', '', term)
    return cleaned.strip()

def normalize_term_name(name):
    """Normalize string for comparison (lower case, strip)."""
    if not isinstance(name, str):
        return ""
    return name.strip().lower()

def load_hpo(hpo_json_path):
    """Load hp.json and return mappings."""
    with open(hpo_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_name = {}
    id_to_synonyms = defaultdict(list)

    if isinstance(data, dict) and "graphs" in data:
        for graph in data["graphs"]:
            nodes = graph.get("nodes", [])
            for node in nodes:
                raw_id = node.get("id")
                term_id = to_hp_id(raw_id)
                if not term_id: continue

                label = node.get("lbl") or node.get("label") or node.get("name")
                if label:
                    id_to_name[term_id] = label

                meta = node.get("meta", {})
                syn_list = meta.get("synonyms", [])
                for syn in syn_list:
                    val = syn.get("val")
                    if isinstance(val, str):
                        id_to_synonyms[term_id].append(val)
    
    return id_to_name, id_to_synonyms

def detect_run_number_from_filename(filename):
    matches = re.findall(r"(\d+)", filename)
    return int(matches[-1]) if matches else None

def validate_csv_file(path, id_to_name, id_to_synonyms, default_run_number=None):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

    def find_col(possible_names):
        target_norms = [normalize_term_name(x) for x in possible_names]
        for col in df.columns:
            if normalize_term_name(col) in target_norms:
                return col
        return None

    hpo_id_col = find_col(["HPO ID", "HPO_ID", "ID", "HPOID"])
    term_name_col = find_col(["Term Name", "Term", "Label", "Name", "Phenotype"])
    run_col = find_col(["Run Number", "Run", "Run_Number"])

    if hpo_id_col is None or term_name_col is None:
        return None

    df = df.rename(columns={hpo_id_col: 'Raw_ID', term_name_col: 'Raw_Term'})
    
    if run_col:
        df = df.rename(columns={run_col: 'Run_Number'})
    else:
        df["Run_Number"] = default_run_number

    is_valid_list = []
    status_list = []
    official_name_list = []
    cleaned_term_list = []
    
    for _, row in df.iterrows():
        raw_id = to_hp_id(str(row['Raw_ID']))
        raw_term = str(row['Raw_Term'])
        
        # --- NEW: Clean the term before matching ---
        clean_term = clean_term_for_matching(raw_term)

        is_valid = False
        status = "Invalid ID"
        official_name = ""

        if raw_id and raw_id in id_to_name:
            is_valid = True
            official_name = id_to_name[raw_id]
            
            # Use cleaned term for normalization
            norm_raw = normalize_term_name(clean_term)
            norm_official = normalize_term_name(official_name)
            
            syns = id_to_synonyms.get(raw_id, [])
            norm_syns = {normalize_term_name(s) for s in syns}

            if norm_raw == norm_official:
                status = "Exact Match"
            elif norm_raw in norm_syns:
                status = "Synonym"
            else:
                status = "Name Mismatch"
        else:
            status = "Invalid ID"

        is_valid_list.append(is_valid)
        status_list.append(status)
        official_name_list.append(official_name)
        cleaned_term_list.append(clean_term)

    df["Official_Term_Name"] = official_name_list
    df["Is_Valid_ID"] = is_valid_list
    df["Status"] = status_list
    df["Cleaned_Term_Used"] = cleaned_term_list # Added for visibility
    df['Normalized_ID'] = df['Raw_ID'].apply(to_hp_id)

    return df

def main():
    print(f"Loading HPO from {HPO_JSON_PATH} ...")
    id_to_name, id_to_synonyms = load_hpo(HPO_JSON_PATH)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    all_dfs = []

    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(".csv") or "validated" in fname:
            continue

        input_path = os.path.join(INPUT_FOLDER, fname)
        run_num = detect_run_number_from_filename(fname)
        model_name = fname.split('_')[0] if '_' in fname else "Unknown"

        df_validated = validate_csv_file(
            input_path, id_to_name, id_to_synonyms, default_run_number=run_num
        )

        if df_validated is not None:
            df_validated['Source_File'] = fname
            df_validated['Model'] = model_name
            all_dfs.append(df_validated)
            print(f"Processed {fname}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_FOLDER, "all_runs_raw_combined.csv"), index=False)
        
        # --- AGGREGATION STEP ---
        print("Aggregating unique terms for review...")
        
        # Group by Raw_Term (original) or Normalized_ID?
        # Usually better to group by ID and keep the *first* raw term observed, 
        # or grouped by ID + Raw Term. 
        # If we group by ID + Raw Term, "Headache (severe)" and "Headache" will be separate rows.
        # This is safer for review so you can see if the parens changed the meaning.
        aggregated = combined.groupby(['Normalized_ID', 'Raw_Term']).agg({
            'Official_Term_Name': 'first',
            'Status': 'first',
            'Is_Valid_ID': 'first',
            'Cleaned_Term_Used': 'first',
            'Run_Number': 'count', 
            'Model': lambda x: ', '.join(sorted(set(x)))
        }).reset_index()

        aggregated.rename(columns={'Run_Number': 'Frequency_Count'}, inplace=True)
        
        # Logic: If Status is Match or Synonym, we trust it.
        aggregated['Ontology_Match_Check'] = aggregated['Status'].isin(['Exact Match', 'Synonym'])

        # --- PRE-FILL SUGGESTED ERROR TYPES ---
        def suggest_error(row):
            if row['Status'] == 'Invalid ID':
                return "Invalid ID"
            elif row['Status'] == 'Name Mismatch':
                return "Check Mapping"
            elif row['Ontology_Match_Check']:
                return "None"
            return "Review"

        aggregated['Suggested_Error_Type'] = aggregated.apply(suggest_error, axis=1)

        # --- USER INPUT COLUMNS ---
        aggregated['MANUAL_Is_Real_Term'] = aggregated['Ontology_Match_Check']
        aggregated['MANUAL_Present_In_Note'] = '' 
        aggregated['MANUAL_Error_Type'] = '' 

        aggregated = aggregated.sort_values(by=['Frequency_Count'], ascending=False)
        review_path = os.path.join(OUTPUT_FOLDER, "aggregated_for_manual_review.csv")
        aggregated.to_csv(review_path, index=False)
        
        print(f"\nSUCCESS: Review file created at: {review_path}")

    else:
        print("No valid CSV files found.")

if __name__ == "__main__":
    main()