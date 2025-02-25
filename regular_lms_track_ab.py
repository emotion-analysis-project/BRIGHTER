#!/usr/bin/env python
import os
import csv
import json
import torch
import numpy as np
import argparse
from typing import List
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
from scipy.stats import pearsonr


def load_data(filepath: str, max_rows: int = None):
    """
    Loads a CSV file for the emotion dataset.
    Expects CSV header: "text", "emotion", plus one column per emotion.
    Returns:
      texts: List of input text strings.
      labels: List of label lists.
      label_names: List of emotion names.
      ids: List of IDs (if present).
    """
    texts = []
    labels = []
    ids = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        possible_emotions = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
        emotion_indices = []
        label_names = []
        text_index = None
        id_index = None

        for idx, col_name in enumerate(header):
            lower = col_name.lower().strip()
            if lower in possible_emotions:
                emotion_indices.append(idx)
                label_names.append(lower)
            elif lower == "text":
                text_index = idx
            elif lower == "id":
                id_index = idx

        row_count = 0
        for row in reader:
            if max_rows is not None and row_count >= max_rows:
                break
            if text_index is not None:
                text_val = row[text_index]
            else:
                text_val = row[1] if len(row) > 1 else ""
            if id_index is not None:
                id_val = row[id_index]
            else:
                id_val = row[0] if len(row) > 0 else f"row_{row_count}"

            row_labels = [int(row[i]) for i in emotion_indices]
            texts.append(text_val)
            labels.append(row_labels)
            ids.append(id_val)
            row_count += 1

    return texts, labels, label_names, ids


class EmotionDataset(Dataset):
    """
    A PyTorch Dataset for emotion classification.
    """
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
        if label is not None:
            # For binary task, labels are float (0/1); for intensity, see below.
            item["labels"] = torch.tensor(label, dtype=torch.float)
        return item


def train_and_evaluate_binary(model_name: str, train_filepath: str, test_filepath: str,
                              max_train_rows: int, num_eval_samples: int,
                              num_epochs: int, lr: float, output_json_path: str):
    """
    Fine-tunes and evaluates a regular LM for binary emotion presence.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    texts_train, labels_train, label_names, _ = load_data(train_filepath, max_rows=max_train_rows)
    num_labels = len(label_names)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.problem_type = "multi_label_classification"  # Binary classification per emotion.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Calculate pos_weight for BCE loss.
    labels_np = np.array(labels_train)
    pos_weights = []
    for i in range(num_labels):
        pos = labels_np[:, i].sum()
        neg = labels_np.shape[0] - pos
        pos_weight = neg / pos if pos > 0 else 1.0
        pos_weights.append(pos_weight)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float)

    train_dataset = EmotionDataset(texts_train, labels_train, tokenizer, max_length=256)

    texts_test, labels_test, label_names_test, _ = load_data(test_filepath, max_rows=num_eval_samples)
    # Reorder test labels to match training label order.
    test_label_map = {name: i for i, name in enumerate(label_names_test)}
    reorder_indices = [test_label_map[ln] for ln in label_names]
    reordered_labels_test = [[row[i] for i in reorder_indices] for row in labels_test]
    eval_dataset = EmotionDataset(texts_test, reordered_labels_test, tokenizer, max_length=256)

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

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred_labels = (probs >= 0.5).astype(int)

        # This returns an array of F1 scores, one per label
        f1_indiv = f1_score(labels, pred_labels, average=None, zero_division=0)
        
        # Now compute the macro average manually
        f1_macro = float(np.mean(f1_indiv))
        
        results = {"f1_macro": f1_macro}
        # Also store individual F1 per emotion
        for i, lbl in enumerate(label_names):
            results[f"f1_{lbl}"] = f1_indiv[i]
        return results

    class CustomTrainer(Trainer):
        def __init__(self, pos_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pos_weights = pos_weights.to(self.model.device)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Binary Evaluation Results:", eval_results)
    # Create the output directory if it does not exist.
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Results saved to {output_json_path}")


def train_and_evaluate_intensity(model_name: str, train_filepath: str, test_filepath: str,
                                 max_train_rows: int, num_eval_samples: int,
                                 num_epochs: int, lr: float, output_json_path: str):
    """
    Fine-tunes and evaluates a regular LM for emotion intensity.
    Here, each emotion's intensity is in {0, 1, 2, 3} and the model outputs (num_emotions*4) logits.
    We override the training step to use CrossEntropyLoss per emotion.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    texts_train, labels_train, label_names, _ = load_data(train_filepath, max_rows=max_train_rows)
    num_emotions = len(label_names)
    # For intensity, we set num_labels to num_emotions*4.
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_emotions * 4
    # Note: We do not set problem_type here because we override the loss.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    train_dataset = EmotionDataset(texts_train, labels_train, tokenizer, max_length=256)

    texts_test, labels_test, label_names_test, _ = load_data(test_filepath, max_rows=num_eval_samples)
    test_label_map = {name: i for i, name in enumerate(label_names_test)}
    reorder_indices = [test_label_map[ln] for ln in label_names]
    reordered_labels_test = [[row[i] for i in reorder_indices] for row in labels_test]
    eval_dataset = EmotionDataset(texts_test, reordered_labels_test, tokenizer, max_length=256)

    training_args = TrainingArguments(
        output_dir="finetuned_model_intensity",
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1000,
        disable_tqdm=False
    )

    def safe_pearsonr(x, y):
        # If all x or y are the same, correlation is undefined => return 0 or skip
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0  # or skip this emotion
        return pearsonr(x, y)[0]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred  # shape: (batch_size, num_emotions)
        batch_size = logits.shape[0]
        
        # Reshape logits => (batch_size, num_emotions, 4)
        logits = logits.reshape(batch_size, num_emotions, 4)
        preds = np.argmax(logits, axis=-1)

        pearson_vals = []
        for i in range(num_emotions):
            r = safe_pearsonr(labels[:, i], preds[:, i])
            pearson_vals.append(r)
        
        results = {"pearsonr_macro_overall": float(np.mean(pearson_vals))}
        for i, lbl in enumerate(label_names):
            results[f"pearsonr_{lbl}"] = float(pearson_vals[i])
        return results


    class CustomTrainerIntensity(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")  # shape: (batch_size, num_emotions)
            outputs = model(**inputs)
            logits = outputs.logits  # shape: (batch_size, num_emotions*4)
            batch_size = logits.shape[0]
            logits = logits.view(batch_size, num_emotions, 4)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 4), labels.view(-1).long())
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainerIntensity(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Intensity Evaluation Results:", eval_results)
    # Create the output directory if it does not exist.
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Results saved to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate regular LMs for emotion tasks.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model from Hugging Face hub")
    parser.add_argument("--task", type=str, choices=["binary", "intensity"], required=True, help="Task type")
    parser.add_argument("--train_filepath", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_filepath", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--max_train_rows", type=int, default=5000, help="Maximum training rows")
    parser.add_argument("--num_eval_samples", type=int, default=5000, help="Number of evaluation samples")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output_json_path", type=str, default="evaluation_results.json", help="Output JSON file for evaluation results")
    
    args = parser.parse_args()

    if args.task == "binary":
        train_and_evaluate_binary(
            model_name=args.model_name,
            train_filepath=args.train_filepath,
            test_filepath=args.test_filepath,
            max_train_rows=args.max_train_rows,
            num_eval_samples=args.num_eval_samples,
            num_epochs=args.num_epochs,
            lr=args.learning_rate,
            output_json_path=args.output_json_path
        )
    else:
        train_and_evaluate_intensity(
            model_name=args.model_name,
            train_filepath=args.train_filepath,
            test_filepath=args.test_filepath,
            max_train_rows=args.max_train_rows,
            num_eval_samples=args.num_eval_samples,
            num_epochs=args.num_epochs,
            lr=args.learning_rate,
            output_json_path=args.output_json_path
        )
