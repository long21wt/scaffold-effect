# The Scaffold Effect: How Prompt Framing Drives Apparent Multimodal Gains in Clinical VLM Evaluation

Code for our paper. We evaluate Vision-Language Models (VLMs) on two clinical classification tasks using multimodal patient data (clinical text + structural MRI).

## Tasks

- **MDD classification** (FOR2107 dataset): Major Depressive Disorder vs. Control
- **Cognitive decline classification** (OASIS dataset): Cognitive Decline vs. Cognitive Normal

## Repository Structure

| File | Description |
|------|-------------|
| `inference.py` | Main inference script for MDD task; supports Gemma-3, LLaVA, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, GLM-4V, InternVL |
| `inference_oasis.py` | Inference script for OASIS cognitive decline task |
| `inference_explain_2.py` | Inference with token-level probability extraction for contrastive analysis |
| `train_dpo.py` | MPO/DPO fine-tuning of Qwen2.5-VL-3B-Instruct on a preference dataset |
| `scaffold_search.py` | Mechanistic interpretability: extracts scaffold direction from layer-33 hidden states and searches for equivalent trigger phrases |
| `summarize_contrastive_2.py` | Aggregates and plots three-way contrastive results (tabular / +preamble / +full MRI) |
| `f1_eval.py` | Computes F1 / precision / recall / accuracy for MDD results |
| `f1_eval_oasis.py` | Same evaluation for OASIS results |

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU (≥24 GB VRAM recommended for 7B+ models).

## Usage

### Inference - MDD (FOR2107)

```bash
python inference.py \
    --txt_path     /path/to/txt_mdd_split/test \
    --mri_base_path /path/to/mri_data \
    --output_file  results_test.jsonl \
    --model_name   Qwen/Qwen2.5-VL-7B-Instruct \
    --mode         tabular_parcel_mri
```

### Inference - OASIS

```bash
python inference_oasis.py \
    --txt_path    /path/to/oasis_txt/test \
    --output_file oasis_results_test.jsonl \
    --model_name  Qwen/Qwen2.5-VL-7B-Instruct \
    --mode        tabular
```

### Evaluation - MDD

```bash
python f1_eval.py \
    --control_file results_control.jsonl \
    --mdd_file     results_mdd.jsonl
```

### Evaluation - OASIS

```bash
python f1_eval_oasis.py \
    --cn_file results_cn.jsonl \
    --cd_file results_cd.jsonl
```

### Contrastive summary & figures

```bash
python summarize_contrastive_2.py \
    --preamble_files results_*_tabular_mri_preamble_*_contrastive.jsonl \
    --full_files     results_*_tabular_parcel_mri_*_contrastive.jsonl \
    --output_dir     ./summary
```

### Fine-tuning (MPO/DPO)

```bash
python train_dpo.py \
    --dataset_dir ./dpo_dataset \
    --output_dir  ./qwen25vl_finetuned \
    --run_name    qwen25vl_mpo
```


### Scaffold direction search (mechanistic interpretability)

```bash
python scaffold_search.py \
    --txt_mdd_path  /path/to/txt_mdd_split/test \
    --txt_ctrl_path /path/to/txt_control_split/test \
    --model_name    Qwen/Qwen2.5-VL-3B-Instruct \
    --n_patients    15 \
    --output_dir    ./scaffold_results
```

## Input Data Format

Each patient is represented by a plain-text file (`<patient_id>.txt`) containing structured clinical features. For MRI conditions, a corresponding folder `sub-<NNNN>/` holds session subfolders with parcellation stats (`.txt`) and brain region visualisation plots (`.png`).

## Output Format

All inference scripts produce `.jsonl` files where each line is a JSON object:

```json
{
  "filename": "0042.txt",
  "mode": "tabular_parcel_mri",
  "output": "{ \"category\": \"Major Depressive Disorder\", \"explanation\": \"...\" }"
}
```

Contrastive mode additionally includes `baseline` and `full` probability fields and a `delta_p_mdd` field.
