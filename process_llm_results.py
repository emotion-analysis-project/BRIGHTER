import os
import glob
import json
import pandas as pd

###############################################################################
# 1. Model Name Mapping & Desired Order
###############################################################################
model_name_map = {
    "final_bothTasks_Qwen_Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "final_bothTasks_databricks_dolly-v2-12b": "Dolly-v2-12B",
    "final_bothTasks_meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "final_bothTasks_mistralai_Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "final_bothTasks_deepseek-ai_DeepSeek-R1-Distill-Llama-70B": "DeepSeek-R1-70B"
}

desired_model_order = [
    "Qwen2.5-72B",
    "Dolly-v2-12B",
    "Llama-3.3-70B",
    "Mixtral-8x7B",
    "DeepSeek-R1-70B"
]

def get_short_model_name_from_data(data):
    """
    Convert the "model_name" string from the JSON file into a short model name.
    We replace "/" with "_" and add the prefix "final_bothTasks_" so that the key
    matches one in our model_name_map. If not found, we return the original string.
    """
    model_full = data.get("model_name", "")
    key = "final_bothTasks_" + model_full.replace("/", "_")
    return model_name_map.get(key, model_full)

###############################################################################
# 2. Utility: Load data
###############################################################################
def load_data(json_path):
    """Return the entire JSON as a dict (or None if load fails)."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception:
        return None

###############################################################################
# 3. Ablation Tables for the Binary Task 
###############################################################################
def collect_prompt_variant_eng(all_files):
    """
    Table 1:
    Rows = [Prompt v1, Prompt v2, Prompt v3]
    Columns = [models in desired order]
    For files where task is 'binary' and language is 'eng', 
    extract the ablation->prompt_variant->v1,v2,v3 (macro_f1) values.
    Multiply by 100 and round.
    """
    rows = ["Prompt v1", "Prompt v2", "Prompt v3"]
    table_data = {}

    for path in all_files:
        data = load_data(path)
        if not data or data.get("task") != "binary":
            continue
        # Only consider English files for this table
        if data.get("language", "").lower() != "eng":
            continue

        model_name = get_short_model_name_from_data(data)
        ablation = data.get("ablation")
        if not ablation:
            continue
        
        prompt_var = ablation.get("prompt_variant", {})
        v1 = prompt_var.get("v1", {}).get("macro_f1", float('nan'))
        v2 = prompt_var.get("v2", {}).get("macro_f1", float('nan'))
        v3 = prompt_var.get("v3", {}).get("macro_f1", float('nan'))
        
        table_data[model_name] = [v1, v2, v3]
    
    df = pd.DataFrame(table_data, index=rows)
    # Reorder columns
    existing_cols = [m for m in desired_model_order if m in df.columns]
    df = df[existing_cols]
    df = df * 100
    df = df.round(2)
    return df

def collect_few_shot_eng(all_files):
    """
    Table 2:
    Rows = [0-shot, 1-shot, 2-shot, 4-shot]
    Columns = [models]
    For files where task is 'binary' and language is 'eng', 
    extract the ablation->few_shot->{0,1,2,4} (macro_f1) values.
    Multiply by 100 and round.
    """
    rows = ["0-shot", "1-shot", "2-shot", "4-shot", "8-shot", "16-shot"]
    table_data = {}

    for path in all_files:
        data = load_data(path)
        if not data or data.get("task") != "binary":
            continue
        if data.get("language", "").lower() != "eng":
            continue

        model_name = get_short_model_name_from_data(data)
        ablation = data.get("ablation")
        if not ablation:
            continue
        
        fs_dict = ablation.get("few_shot", {})
        val_0 = fs_dict.get("0", {}).get("macro_f1", float('nan'))
        val_1 = fs_dict.get("1", {}).get("macro_f1", float('nan'))
        val_2 = fs_dict.get("2", {}).get("macro_f1", float('nan'))
        val_4 = fs_dict.get("4", {}).get("macro_f1", float('nan'))
        val_8 = fs_dict.get("8", {}).get("macro_f1", float('nan'))
        val_16 = fs_dict.get("16", {}).get("macro_f1", float('nan'))
        
        table_data[model_name] = [val_0, val_1, val_2, val_4, val_8, val_16]
    
    df = pd.DataFrame(table_data, index=rows)
    existing_cols = [m for m in desired_model_order if m in df.columns]
    df = df[existing_cols]
    df = df * 100
    df = df.round(2)
    return df

def collect_top_k_eng(all_files):
    """
    Table 3:
    Rows = [@1, @2, @4]
    Columns = [models]
    For files where task is 'binary' and language is 'eng', 
    extract the ablation->top_k->{1,2,4} (macro_f1) values.
    Multiply by 100 and round.
    """
    rows = ["@1", "@2", "@4", "@8"]
    table_data = {}

    for path in all_files:
        data = load_data(path)
        if not data or data.get("task") != "binary":
            continue
        if data.get("language", "").lower() != "eng":
            continue

        model_name = get_short_model_name_from_data(data)
        ablation = data.get("ablation")
        if not ablation:
            continue
        
        topk_dict = ablation.get("top_k", {})
        val_1 = topk_dict.get("1", {}).get("macro_f1", float('nan'))
        val_2 = topk_dict.get("2", {}).get("macro_f1", float('nan'))
        val_4 = topk_dict.get("4", {}).get("macro_f1", float('nan'))
        val_8 = topk_dict.get("8", {}).get("macro_f1", float('nan'))
        
        table_data[model_name] = [val_1, val_2, val_4, val_8]
    
    df = pd.DataFrame(table_data, index=rows)
    existing_cols = [m for m in desired_model_order if m in df.columns]
    df = df[existing_cols]
    df = df * 100
    df = df.round(2)
    return df

def collect_english_vs_native(all_files):
    """
    Table 4:
    Rows = multi-index of (Language, 'Prompt in eng') and (Language, 'Prompt in {lang}')
    Columns = [models]
    For each file (with task 'binary') that has non-null ablation->english_v1_vs_native_v1,
    extract:
      - f1_english_v1: the result when the prompt is in English,
      - f1_native_v1: the result when the prompt is in the target (native) language.
    (Both values are taken from the 'macro_f1' field.)
    """
    all_rows = set()  # Will hold tuples of (language, variant label)
    data_by_model = {}

    for path in all_files:
        data = load_data(path)
        if not data or data.get("task") != "binary":
            continue

        model_name = get_short_model_name_from_data(data)
        language = data.get("language", "")
        ablation = data.get("ablation")
        if not ablation:
            continue

        eng_vs_nat = ablation.get("english_v1_vs_native_v1")
        if not eng_vs_nat:
            continue

        f1_eng = eng_vs_nat.get("f1_english_v1", {}).get("macro_f1", float('nan'))
        f1_nat = eng_vs_nat.get("f1_native_v1", {}).get("macro_f1", float('nan'))

        if model_name not in data_by_model:
            data_by_model[model_name] = {}

        # The file's language is the target language.
        # We record both the result when using an English prompt and the result
        # when using a prompt in the target language.
        data_by_model[model_name][(language, "Prompt in eng")] = f1_eng
        data_by_model[model_name][(language, f"Prompt in {language}")] = f1_nat

        all_rows.add((language, "Prompt in eng"))
        all_rows.add((language, f"Prompt in {language}"))

    # Sort the row keys (first by language)
    sorted_rows = sorted(all_rows, key=lambda x: x[0])
    model_names = [m for m in desired_model_order if m in data_by_model]

    # Build table matrix
    table_values = []
    for row in sorted_rows:
        row_vals = []
        for m in model_names:
            val = data_by_model[m].get(row, float('nan'))
            row_vals.append(val)
        table_values.append(row_vals)

    idx = pd.MultiIndex.from_tuples(sorted_rows, names=["Language", "Ablation"])
    df = pd.DataFrame(table_values, index=idx, columns=model_names)
    df = df * 100
    df = df.round(2)
    return df

###############################################################################
# 4. Main Results Tables for 'binary' & 'intensity'
###############################################################################
def collect_main_results(all_files, task="binary"):
    """
    For each file whose task matches (binary or intensity),
    extract the main_result->macro_f1 value. (Each file is for one language.)
    Build a table with rows = languages and columns = models.
    Multiply by 100 and round.
    """
    all_langs = set()
    data_by_model = {}

    for path in all_files:
        data = load_data(path)
        if not data or data.get("task") != task:
            continue

        model_name = get_short_model_name_from_data(data)
        language = data.get("language", "")
        main_result = data.get("main_result", {})
        if task == "intensity":
            mr = main_result.get("avg_pearson", float('nan'))
        else:
            mr = main_result.get("macro_f1", float('nan'))

        if model_name not in data_by_model:
            data_by_model[model_name] = {}
        data_by_model[model_name][language] = mr
        all_langs.add(language)

    sorted_langs = sorted(all_langs)
    sorted_models = [m for m in desired_model_order if m in data_by_model]

    # Build a table: rows = languages, columns = models
    matrix = []
    for lang in sorted_langs:
        row = []
        for model in sorted_models:
            val = data_by_model[model].get(lang, float('nan'))
            row.append(val)
        matrix.append(row)
    
    df = pd.DataFrame(matrix, index=sorted_langs, columns=sorted_models)
    df = df * 100
    df = df.round(2)
    return df

###############################################################################
# 5. main()
###############################################################################
def main():
    # 1. Gather all JSON files from the results folder.
    all_files = glob.glob("llm_track_ab_results/*.json")

    # 2. Construct ablation tables for the binary task.
    df_prompt_variant = collect_prompt_variant_eng(all_files)
    df_few_shot = collect_few_shot_eng(all_files)
    df_top_k = collect_top_k_eng(all_files)
    df_eng_vs_native = collect_english_vs_native(all_files)

    # 3. Construct main results tables.
    df_main_binary = collect_main_results(all_files, task="binary")
    df_main_intensity = collect_main_results(all_files, task="intensity")

    # 4. Output ablation tables (CSV + LaTeX)
    df_prompt_variant.to_csv("llm_track_ab_results/table_prompt_variant.csv", float_format="%.2f")
    df_few_shot.to_csv("llm_track_ab_results/table_few_shot.csv", float_format="%.2f")
    df_top_k.to_csv("llm_track_ab_results/table_top_k.csv", float_format="%.2f")
    df_eng_vs_native.to_csv("llm_track_ab_results/table_english_vs_native.csv", float_format="%.2f")

    with open("llm_track_ab_results/table_prompt_variant.tex", "w") as f:
        f.write(df_prompt_variant.to_latex(float_format="%.2f"))
    with open("llm_track_ab_results/table_few_shot.tex", "w") as f:
        f.write(df_few_shot.to_latex(float_format="%.2f"))
    with open("llm_track_ab_results/table_top_k.tex", "w") as f:
        f.write(df_top_k.to_latex(float_format="%.2f"))
    with open("llm_track_ab_results/table_english_vs_native.tex", "w") as f:
        f.write(df_eng_vs_native.to_latex(float_format="%.2f", multirow=True))

    # 5. Output main results tables (CSV + LaTeX)
    df_main_binary.to_csv("llm_track_ab_results/table_main_binary.csv", float_format="%.2f")
    df_main_intensity.to_csv("llm_track_ab_results/table_main_intensity.csv", float_format="%.2f")

    with open("llm_track_ab_results/table_main_binary.tex", "w") as f:
        f.write(df_main_binary.to_latex(float_format="%.2f"))
    with open("llm_track_ab_results/table_main_intensity.tex", "w") as f:
        f.write(df_main_intensity.to_latex(float_format="%.2f"))

    print("All tables have been saved to CSV and LaTeX files in llm_track_ab_results/.")

if __name__ == "__main__":
    main()