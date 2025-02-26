# README: Multilingual Emotion Classification with Language Models

> The official results can be found on [Google Drive](https://drive.google.com/drive/folders/1cu3ucbNp-f8X5G98wOw_IFjTWGwpoy4u?usp=sharing).

## Overview

This repository contains scripts for training, evaluating, and analyzing language models (LMs) for multilingual emotion classification. The experiments include both fine-tuned transformer models and large language models (LLMs) evaluated in a zero-shot or few-shot setting. The project supports:

- **Emotion Classification** (Track A): Detecting the presence/absence of six emotions (anger, disgust, fear, joy, sadness, surprise).
- **Emotion Intensity Classification** (Track B): Predicting the intensity of emotions on a scale of 0 (none) to 3 (high).
- **Multilingual Emotion Classification** (Track C): Evaluation on cross-lingual transfer scenarios.

## Repository Structure

```
.
├── regular_lms_track_c.py        # Fine-tuning transformers (Track C: Leave-One-Out on Language Families)
├── regular_lms_track_ab.py       # Fine-tuning transformers (Track A/B: Train on one language, test on same)
├── process_lm_results.py         # Processes results for fine-tuned transformer models
├── regular_lms_track_c_array.sh  # SLURM script for Track C experiments
├── regular_lms_track_ab_array.sh # SLURM script for Track A/B experiments
├── process_llm_results.py        # Processes results for large language models (LLMs)
├── llms.py                       # Runs inference with LLMs (e.g., LLaMA, Mixtral, DeepSeek)
├── llms_array.sh                 # SLURM script for LLM evaluations
├── track_a/                      # Data for Track A (Binary Classification)
│   ├── train/                    # Training CSV files (one per language)
│   ├── test/                     # Test CSV files (one per language)
├── track_b/                      # Data for Track B (Intensity Classification)
│   ├── train/                    # Training CSV files (one per language)
│   ├── test/                     # Test CSV files (one per language)
├── lm_track_ab_results/          # Output directory for fine-tuned LM results (Track A/B)
├── lm_track_c_results/           # Output directory for fine-tuned LM results (Track C)
├── llm_track_ab_results/         # Output directory for LLM evaluation results
└── README.md                     # This file
```

## 1. Fine-Tuned Transformer Models (Track A, B, C)

### Training & Evaluation

For **fine-tuning transformer models**, we use `regular_lms_track_ab.py` and `regular_lms_track_c.py`.

#### **Track A/B: Fine-tuning on a single language**
- Train on a given language and evaluate on the same language.
- Supports binary emotion classification and intensity classification.

**Run Track A/B fine-tuning:**
```bash
uv run python regular_lms_track_ab.py \
    --model_name "facebook/xlm-roberta-large" \
    --task "binary" \
    --train_filepath "./track_a/train/eng.csv" \
    --test_filepath "./track_a/test/eng.csv" \
    --output_json_path "./lm_track_ab_results/xlm-roberta-large_binary/eng_results.json"
```

#### **Track C: Family-Based Leave-One-Out**
- Train on all languages in a family except one, and test on the held-out language.

**Run Track C fine-tuning:**
```bash
uv run python regular_lms_track_c.py \
    --model_name "facebook/xlm-roberta-large" \
    --train_dir "./track_a/train" \
    --test_dir "./track_a/test" \
    --output_dir "./lm_track_c_results" \
    --num_epochs 2 \
    --learning_rate 1e-5
```

#### **Run fine-tuning on SLURM (GPU Cluster)**
```bash
sbatch regular_lms_track_ab_array.sh   # For Track A/B
sbatch regular_lms_track_c_array.sh    # For Track C
```

### Processing Fine-Tuned Model Results

```bash
uv run python process_lm_results.py
```
This script:
- Aggregates results for Track A, B, and C.
- Generates CSV tables and LaTeX tables for analysis.

## 2. Large Language Models (LLMs) - Zero/Few-Shot Evaluation

### LLM Evaluation Script (`llms.py`)

Instead of fine-tuning, `llms.py` evaluates LLMs (like LLaMA, Mixtral, DeepSeek) in a zero-shot or few-shot setting as well as ablations regarding number of few-shot examples, topk, and prompts in English vs the native languages.

#### **Run LLM Evaluation:**
```bash
uv run python llms.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --task "binary" \
    --tensor_parallel_size 8 \
    --output_file "./llm_track_ab_results/llama3_70B_binary_eng.json"
```

#### **Run LLM Evaluation on SLURM:**
```bash
sbatch llms_array.sh
```
This script evaluates all LLMs in parallel across multiple GPUs.

### Processing LLM Results

```bash
uv run python process_llm_results.py
```
This script:
- Aggregates results for binary and intensity tasks.
- Generates tables for different ablation studies (prompt variants, few-shot, top-k sampling).

## 3. Data Format

### CSV File Format (for fine-tuning)

Each language dataset is stored as a CSV file with the following columns:

| id  | text                                | anger | disgust | fear | joy | sadness | surprise |
|-----|-------------------------------------|-------|---------|------|-----|---------|----------|
| 1   | "I'm so happy today!"               | 0     | 0       | 0    | 1   | 0       | 0        |
| 2   | "This is so frustrating!"           | 1     | 0       | 0    | 0   | 0       | 0        |

- **Binary Task:** Labels are `1` (emotion present) or `0` (emotion absent).
- **Intensity Task:** Labels range from `0` (none) to `3` (high).

## 4. Evaluation Metrics

- **Binary Classification**: **Macro-F1 score** (averaged across emotions)
- **Intensity Classification**: **Pearson correlation** between predicted and true intensity values.

## 5. Supported Models

### **Fine-Tuned Transformer Models**
In principle any autoencoder model for SequenceClassification from HuggingFace should work, but the ones below are tested.

| Model                      | Hugging Face Model ID                        |
|----------------------------|---------------------------------------------|
| XLM-R Large                | `facebook/xlm-roberta-large`               |
| mBERT                      | `google-bert/bert-base-multilingual-cased` |
| RemBERT                    | `google/rembert`                           |
| InfoXLM                    | `microsoft/infoxlm-large`                  |
| mDeBERTa                   | `microsoft/mdeberta-v3-base`               |
| LaBSE                      | `sentence-transformers/LaBSE`              |

### **Large Language Models (Zero/Few-Shot)**
In principle any autoregressive LLM from HuggingFace should work, but the ones below are tested.

| Model                      | Hugging Face Model ID                           |
|----------------------------|------------------------------------------------|
| LLaMA 3.3 70B              | `meta-llama/Llama-3.3-70B-Instruct`            |
| Mixtral 8x7B               | `mistralai/Mixtral-8x7B-Instruct-v0.1`         |
| DeepSeek R1 70B            | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`    |
| Qwen 2.5 72B               | `Qwen/Qwen2.5-72B-Instruct`                    |
| Dolly v2 12B               | `databricks/dolly-v2-12b`                      |

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{muhammad2025brighterbridginggaphumanannotated,
      title={BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages}, 
      author={Shamsuddeen Hassan Muhammad and Nedjma Ousidhoum and Idris Abdulmumin and Jan Philip Wahle and Terry Ruas and Meriem Beloucif and Christine de Kock and Nirmal Surange and Daniela Teodorescu and Ibrahim Said Ahmad and David Ifeoluwa Adelani and Alham Fikri Aji and Felermino D. M. A. Ali and Ilseyar Alimova and Vladimir Araujo and Nikolay Babakov and Naomi Baes and Ana-Maria Bucur and Andiswa Bukula and Guanqun Cao and Rodrigo Tufino Cardenas and Rendi Chevi and Chiamaka Ijeoma Chukwuneke and Alexandra Ciobotaru and Daryna Dementieva and Murja Sani Gadanya and Robert Geislinger and Bela Gipp and Oumaima Hourrane and Oana Ignat and Falalu Ibrahim Lawan and Rooweither Mabuya and Rahmad Mahendra and Vukosi Marivate and Andrew Piper and Alexander Panchenko and Charles Henrique Porto Ferreira and Vitaly Protasov and Samuel Rutunda and Manish Shrivastava and Aura Cristina Udrea and Lilian Diana Awuor Wanzare and Sophie Wu and Florian Valentin Wunderlich and Hanif Muhammad Zhafran and Tianhui Zhang and Yi Zhou and Saif M. Mohammad},
      year={2025},
      eprint={2502.11926},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11926}, 
}
```
