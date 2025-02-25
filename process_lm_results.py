import os
import json
import csv
from collections import defaultdict

# A mapping from the base model folder names to a common short form.
model_mapping = {
    "FacebookAI_xlm-roberta-large": "XLM-R",
    "google-bert_bert-base-multilingual-cased": "mBERT",
    "google_rembert": "RemBERT",
    "microsoft_infoxlm-large": "InfoXLM",
    "microsoft_mdeberta-v3-base": "mDeBERTa",
    "sentence-transformers_LaBSE": "LaBSE"
}

# Prepare data structures: one each for binary, intensity, and track C.
table_data = {
    'binary': defaultdict(dict),
    'intensity': defaultdict(dict),
    'track_c': defaultdict(dict),
}

# Folders for track A/B and track C
DEFAULT_FOLDER_AB = './lm_track_ab_results'
DEFAULT_FOLDER_C = './lm_track_c_results'

# ------------------------------------------------------------------------------
# 1. Gather track A/B data (Mostly unchanged, except for the key for "intensity")
# ------------------------------------------------------------------------------
for folder in os.listdir(DEFAULT_FOLDER_AB):
    full_path = os.path.join(DEFAULT_FOLDER_AB, folder)
    if os.path.isdir(full_path):
        # Distinguish between _binary and _intensity
        if folder.endswith('_binary') or folder.endswith('_intensity'):
            eval_type = 'binary' if folder.endswith('_binary') else 'intensity'
            
            # Remove the trailing '_binary' or '_intensity' to get the base model name
            base_model = folder.rsplit('_', 1)[0]
            common_short = model_mapping.get(base_model, base_model)

            # Inside each model folder, each subfolder is a language code
            for lang in os.listdir(full_path):
                lang_dir = os.path.join(full_path, lang)
                if os.path.isdir(lang_dir):
                    # Expect a file called "<lang>_results.json" inside
                    json_file = os.path.join(lang_dir, f"{lang}_results.json")
                    if os.path.exists(json_file):
                        with open(json_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        # For binary: "eval_f1_macro"
                        # For intensity: "eval_pearsonr_macro_overall" (as per your code)
                        if eval_type == 'binary':
                            score = results.get("eval_f1_macro")
                        else:
                            score = results.get("eval_pearsonr_macro_overall")

                        if score is not None:
                            table_data[eval_type][lang][common_short] = score

# ---------------------------------------------------------------------------
# 2. Gather track C data using the *3-level* folder structure from your tree
# ---------------------------------------------------------------------------
if os.path.isdir(DEFAULT_FOLDER_C):
    for model_folder in os.listdir(DEFAULT_FOLDER_C):
        model_path = os.path.join(DEFAULT_FOLDER_C, model_folder)
        if os.path.isdir(model_path):
            # e.g. "FacebookAI_xlm-roberta-large"
            base_model = model_folder
            common_short = model_mapping.get(base_model, base_model)

            # Each subfolder is something like "AfroAsiatic", "Bantu", "SinoTibetan_slavic_train", etc.
            for family_folder in os.listdir(model_path):
                family_path = os.path.join(model_path, family_folder)
                if os.path.isdir(family_path):
                    # Next level might be: "holdout_amh", "holdout_arq", etc.
                    # OR it might directly have a single *.json file (like "chn_results.json").
                    for item in os.listdir(family_path):
                        item_path = os.path.join(family_path, item)

                        # Case 1: We have a subfolder named "holdout_<lang>"
                        if os.path.isdir(item_path) and item.startswith("holdout_"):
                            lang_code = item.replace("holdout_", "")
                            json_file = os.path.join(item_path, f"{lang_code}_results.json")
                            if os.path.exists(json_file):
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    results = json.load(f)
                                # The key for Track C metric, e.g. "eval_f1_macro"
                                score = results.get("eval_f1_macro", None)
                                if score is not None:
                                    table_data['track_c'][lang_code][common_short] = score

                        # Case 2: We have a single JSON file directly in the family folder
                        elif item.endswith("_results.json") and os.path.isfile(item_path):
                            # e.g., "chn_results.json" or "tat_results.json"
                            lang_code = item.replace("_results.json", "")
                            with open(item_path, 'r', encoding='utf-8') as f:
                                results = json.load(f)
                            score = results.get("eval_f1_macro", None)
                            if score is not None:
                                table_data['track_c'][lang_code][common_short] = score
else:
    print(f"Track C folder not found at {DEFAULT_FOLDER_C}.")

# --------------------------------------------------------------
# Helper functions to generate CSV and LaTeX (mostly unchanged)
# --------------------------------------------------------------
def generate_csv(table, filename):
    """
    table is a dict: table[lang][model_short] = float
    """
    # Remove InfoXLM from mdoels
    for lang in table:
        table[lang].pop("InfoXLM", None)

    # Collect all language codes (sorted) and all model columns (union across languages).
    languages = sorted(table.keys())
    model_columns = set()
    for lang in languages:
        model_columns.update(table[lang].keys())
    model_columns = sorted(model_columns)

    header = ["Language"] + model_columns

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for lang in languages:
            row = [lang]
            for model in model_columns:
                score = table[lang].get(model)
                # Multiply by 100 if you want a percentage
                score = score * 100 if score is not None else None
                cell = f"{score:.2f}" if score is not None else ""
                row.append(cell)
            writer.writerow(row)
    print(f"CSV table written to {filename}")

def generate_latex(table, filename):
    """
    table is a dict: table[lang][model_short] = float
    """
    languages = sorted(table.keys())
    model_columns = set()
    for lang in languages:
        model_columns.update(table[lang].keys())
    model_columns = sorted(model_columns)

    with open(filename, 'w', encoding='utf-8') as f:
        # Begin the LaTeX tabular
        f.write("\\begin{tabular}{" + "l" + "c" * len(model_columns) + "}\n")
        # Header row
        f.write("\\toprule\n")
        header_line = "Language & " + " & ".join(model_columns) + " \\\\\n"
        f.write(header_line)
        f.write("\\midrule\n")

        for lang in languages:
            cells = [lang]
            for model in model_columns:
                score = table[lang].get(model)
                score = score * 100 if score is not None else None
                cell = f"{score:.2f}" if score is not None else ""
                cells.append(cell)
            row_line = " & ".join(cells) + " \\\\\n"
            f.write(row_line)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"LaTeX table written to {filename}")

# --------------------------------------------------------------------
# 3. Generate CSV/LaTeX for each evaluation type (unchanged)
# --------------------------------------------------------------------
os.makedirs(DEFAULT_FOLDER_AB, exist_ok=True)
os.makedirs(DEFAULT_FOLDER_C, exist_ok=True)

# Track A/B: binary
generate_csv(table_data['binary'], os.path.join(DEFAULT_FOLDER_AB, "track_a_table.csv"))
generate_latex(table_data['binary'], os.path.join(DEFAULT_FOLDER_AB, "track_a_table.tex"))

# Track A/B: intensity
generate_csv(table_data['intensity'], os.path.join(DEFAULT_FOLDER_AB, "track_b_table.csv"))
generate_latex(table_data['intensity'], os.path.join(DEFAULT_FOLDER_AB, "track_b_table.tex"))

# Track C
generate_csv(table_data['track_c'], os.path.join(DEFAULT_FOLDER_C, "track_c_table.csv"))
generate_latex(table_data['track_c'], os.path.join(DEFAULT_FOLDER_C, "track_c_table.tex"))
