#!/usr/bin/env python
import os
import csv
import json
import torch
import numpy as np
import argparse
from typing import List, Dict
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import wandb
wandb.init(mode="disabled")

###############################################################################
# 1. Families and languages
###############################################################################
FAMILIES = {
    # 7 AfroAsiatic languages:
    "AfroAsiatic": ["amh", "arq", "ary", "hau", "orm", "som", "tir"],
    
    # 1 Austronesian language:
    "Austronesian": ["sun", "ind", "jav"],
    
    # 4 Germanic languages:
    "Germanic": ["afr", "deu", "eng", "swe"],
    
    # 2 IndoIranian languages:
    "IndoIranian": ["hin", "mar"],
    
    # 3 Romance languages:
    "Romance": ["esp", "ptbr", "ptmz", "ron"],  
    # (Replacing "spa"->"esp" and "pt_br"/"pt_mz"->"ptbr"/"ptmz")
    
    # 2 Slavic languages:
    "Slavic": ["rus", "ukr"],
    
    # 1 Turkic language:
    "Turkic": ["tat"],
    
    # 3 Bantu languages:
    "Bantu": ["kin", "swa", "vmw"],
    
    # 2 NigerCongo languages:
    "NigerCongo": ["ibo", "yor", "zul", "xho"],
    
    # 1 Creole language:
    "Creole": ["pcm"],
    
    # SinoTibetan (single language):
    "SinoTibetan": ["chn"]
}

###############################################################################
# 2. Data utilities
###############################################################################
def load_data(filepath: str):
    """
    Loads CSV data, always using the same fixed EMOTION_ORDER. 
    If the CSV is missing any emotion column, assigns label=0 for that emotion.
    """
    EMOTION_ORDER = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    texts = []
    labels = []
    ids = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # read the header row

        # Find which column is 'text', which is 'id'
        text_index = None
        id_index = None
        # For emotions, we'll keep a dict from emotion -> column index if it exists
        emotion_to_col_idx = {}

        for i, col_name in enumerate(header):
            col_lower = col_name.lower().strip()
            if col_lower == "text":
                text_index = i
            elif col_lower == "id":
                id_index = i
            elif col_lower in EMOTION_ORDER:
                # We'll store the column index for this emotion
                emotion_to_col_idx[col_lower] = i

        # For rows
        for row_count, row in enumerate(reader):
            # Get text
            if text_index is not None and text_index < len(row):
                text_val = row[text_index]
            else:
                text_val = ""

            # Get ID
            if id_index is not None and id_index < len(row):
                id_val = row[id_index]
            else:
                id_val = f"row_{row_count}"

            # Build the label array, in the fixed EMOTION_ORDER
            row_labels = []
            for emo in EMOTION_ORDER:
                if emo in emotion_to_col_idx and emotion_to_col_idx[emo] < len(row):
                    row_labels.append(int(row[emotion_to_col_idx[emo]]))
                else:
                    # If this emotion doesn't exist in CSV, assign 0 by default
                    row_labels.append(0)

            texts.append(text_val)
            labels.append(row_labels)
            ids.append(id_val)

    return texts, labels, EMOTION_ORDER, ids


class EmotionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item

###############################################################################
# 3. Fine-tuning / evaluation for binary classification
###############################################################################
def train_and_evaluate_binary(
    model_name: str,
    train_texts: List[str],
    train_labels: List[List[int]],
    test_texts: List[str],
    test_labels: List[List[int]],
    label_names: List[str],
    output_json_path: str,
    num_epochs: int=2,
    lr: float=1e-5,
    skip_exist=True
):
    """
    Fine-tune and evaluate a model for binary classification of multiple emotions.
    """
    if skip_exist and os.path.exists(output_json_path):
        print(f"Skipping existing output file: {output_json_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    num_labels = len(label_names)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Compute pos_weights for BCEWithLogitsLoss
    labels_np = np.array(train_labels)
    pos_weights = []
    for i in range(num_labels):
        pos = labels_np[:, i].sum()
        neg = labels_np.shape[0] - pos
        pos_weight = (neg / pos) if pos > 0 else 1.0
        pos_weights.append(pos_weight)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float)

    # Build datasets
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length=256)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length=256)

    # Training args
    training_args = TrainingArguments(
        output_dir="finetuned_model_binary",
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1000,
        disable_tqdm=False
    )

    # Metric
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Sigmoid then threshold at 0.5
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

        # Macro-F1 across all emotions
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        return {"f1_macro": f1_macro}

    # Custom Trainer to handle pos_weight
    class CustomTrainer(Trainer):
        def __init__(self, pos_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pos_weights = pos_weights.to(self.model.device)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        pos_weights=pos_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    eval_results = trainer.evaluate()

    print(f"Saving results to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

###############################################################################
# 4. Main logic: train on all-langs-in-family-except-one, test on the left-out lang
###############################################################################
def gather_data_for_languages(
    lang_list: List[str],
    train_dir: str,
    test_dir: str
):
    """
    Given a list of languages (their codes), load all texts/labels from the CSVs
    in `train_dir` and `test_dir`, *concatenate* them. 
    Returns: (texts, labels, label_names).
    NOTE: With the updated load_data(), label_names is always the same fixed set:
          ["anger", "disgust", "fear", "joy", "sadness", "surprise"].
    """
    all_texts = []
    all_labels = []
    label_names_master = None

    for lang in lang_list:
        train_fp = os.path.join(test_dir, f"{lang}.csv")
        t_texts, t_labels, label_names, _ = load_data(train_fp)

        if label_names_master is None:
            label_names_master = label_names
        else:
            # Just sanity-check that the new label_names matches
            if label_names != label_names_master:
                raise ValueError(f"Label mismatch between languages {lang_list[0]} and {lang}")

        all_texts.extend(t_texts)
        all_labels.extend(t_labels)

    return all_texts, all_labels, label_names_master


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default="./track_a/train")
    parser.add_argument("--test_dir", type=str, default="./track_a/test")
    parser.add_argument("--output_dir", type=str, default="./family_leave_one_out_results")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    args = parser.parse_args()

    model_name = args.model_name
    train_dir = args.train_dir
    test_dir = args.test_dir
    out_dir_base = args.output_dir
    num_epochs = args.num_epochs
    lr = args.learning_rate

    # For each family with multiple languages, do leave-one-out
    for family_name, langs in FAMILIES.items():

        if len(langs) > 1:
            # We have multiple languages.  For each "held-out" language:
            for held_out_lang in langs:
                # The training set is all the other languages in that family
                train_langs = [l for l in langs if l != held_out_lang]

                # Load training data (concatenate from all train_langs)
                train_texts, train_labels, label_names = gather_data_for_languages(train_langs, train_dir, test_dir)

                # Load test data from the single held-out language
                test_file = os.path.join(test_dir, f"{held_out_lang}.csv")
                test_texts, test_labels, test_label_names, _ = load_data(test_file)

                # *No reorder needed now*, because load_data always uses the same fixed EMOTION_ORDER
                # but for safety, confirm they're the same length
                if len(test_label_names) != len(label_names):
                    raise ValueError("Mismatch in label count between training and test.")
                
                out_subdir = os.path.join(
                    out_dir_base,
                    model_name.replace("/", "_"),
                    f"{family_name}",
                    f"holdout_{held_out_lang}"
                )
                os.makedirs(out_subdir, exist_ok=True)
                out_json = os.path.join(out_subdir, f"{held_out_lang}_results.json")

                print("=====================================================")
                print(f"FAMILY: {family_name}")
                print(f"Training on: {train_langs}")
                print(f"Testing on:  {held_out_lang}")
                print("=====================================================")

                train_and_evaluate_binary(
                    model_name=model_name,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    label_names=label_names,
                    output_json_path=out_json,
                    num_epochs=num_epochs,
                    lr=lr
                )

        else:
            # Family has only one language
            solo_lang = langs[0]
            # If it is Tatar, train on Slavic, test on Tatar
            if solo_lang == "tat":
                slavic_langs = FAMILIES["Slavic"]
                train_texts, train_labels, label_names = gather_data_for_languages(slavic_langs, train_dir, test_dir)

                test_file = os.path.join(test_dir, "tat.csv")
                test_texts, test_labels, test_label_names, _ = load_data(test_file)

                out_subdir = os.path.join(
                    out_dir_base,
                    model_name.replace("/", "_"),
                    "Turkic_slavic_train"
                )
                os.makedirs(out_subdir, exist_ok=True)
                out_json = os.path.join(out_subdir, "tat_results.json")

                print("=====================================================")
                print("SPECIAL CASE: TATAR => train on Slavic, test on Tatar")
                print(f"Training on Slavic: {slavic_langs}")
                print(f"Testing on Tatar")
                print("=====================================================")

                train_and_evaluate_binary(
                    model_name=model_name,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    label_names=label_names,
                    output_json_path=out_json,
                    num_epochs=num_epochs,
                    lr=lr
                )

            # If it is Pidgin, train on Niger-Congo, test on Pidgin
            elif solo_lang == "pcm":
                niger_congo_langs = FAMILIES["NigerCongo"]
                train_texts, train_labels, label_names = gather_data_for_languages(niger_congo_langs, train_dir, test_dir)

                test_file = os.path.join(test_dir, "pcm.csv")
                test_texts, test_labels, test_label_names, _ = load_data(test_file)

                out_subdir = os.path.join(
                    out_dir_base,
                    model_name.replace("/", "_"),
                    "Creole_nigercongo_train"
                )
                os.makedirs(out_subdir, exist_ok=True)
                out_json = os.path.join(out_subdir, "pcm_results.json")

                print("=====================================================")
                print("SPECIAL CASE: PIDGIN => train on Niger-Congo, test on Pidgin")
                print(f"Training on Niger-Congo: {niger_congo_langs}")
                print(f"Testing on Pidgin")
                print("=====================================================")

                train_and_evaluate_binary(
                    model_name=model_name,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    label_names=label_names,
                    output_json_path=out_json,
                    num_epochs=num_epochs,
                    lr=lr
                )

            # If it is Chinese (chn), train on Slavic, test on chn
            elif solo_lang == "chn":
                slavic_langs = FAMILIES["Slavic"]
                train_texts, train_labels, label_names = gather_data_for_languages(slavic_langs, train_dir, test_dir)

                test_file = os.path.join(test_dir, "chn.csv")
                test_texts, test_labels, test_label_names, _ = load_data(test_file)

                out_subdir = os.path.join(
                    out_dir_base,
                    model_name.replace("/", "_"),
                    "SinoTibetan_slavic_train"
                )
                os.makedirs(out_subdir, exist_ok=True)
                out_json = os.path.join(out_subdir, "chn_results.json")

                print("=====================================================")
                print("SPECIAL CASE: CHINESE => train on Slavic, test on chn")
                print(f"Training on Slavic: {slavic_langs}")
                print(f"Testing on Chinese (chn)")
                print("=====================================================")

                train_and_evaluate_binary(
                    model_name=model_name,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    test_texts=test_texts,
                    test_labels=test_labels,
                    label_names=label_names,
                    output_json_path=out_json,
                    num_epochs=num_epochs,
                    lr=lr
                )

            else:
                # If you want to skip other single-language families, do it here:
                print(f"Skipping single-language family: {family_name} ({solo_lang})")


if __name__ == "__main__":
    main()
