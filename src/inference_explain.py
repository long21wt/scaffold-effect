"""
MDD inference pipeline with contrastive explanation support for Qwen2.5-VL and Ministral.
Runs classification across train/dev/test splits and optionally produces contrastive outputs.

Modes:
  - tabular:              text data only (baseline)
  - tabular_mri_preamble: text + MRI preamble string (ablation, no actual MRI images)
  - tabular_parcel_mri:   text + parcel text + MRI images (full multimodal)

Usage:
    python inference_explain.py \
        --txt_mdd_base_path /path/to/txt_mdd_split \
        --model_name        Qwen/Qwen2.5-VL-3B-Instruct \
        --mode              tabular_parcel_mri \
        --mri_mdd_base_path /path/to/mri_mdd \
        --output_dir        ./results \
        --run_contrastive
"""

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    set_seed,
)


@dataclass
class InferenceConfig:
    txt_mdd_path: str  # e.g. .../txt_mdd_split/train
    mri_mdd_path: Optional[str] = None  # e.g. .../mri_mdd_split/train
    output_file: str = "results.jsonl"
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 4096
    do_sample: bool = False
    categories: List[str] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = [
                "Major Depressive Disorder",
                "Control (no disorder detected)",
            ]


def find_category_token_idx(generated_tokens: List[str]) -> Optional[int]:
    """
    Robustly find the index of the label token ('Major' or 'Control')
    that is the value of the 'category' JSON key.

    Strategy: scan for the 'category' key then take the first label
    token that appears after it — ignoring any later occurrences inside
    the explanation field.
    """
    category_key_idx = None
    for i, token in enumerate(generated_tokens):
        if "category" in token.lower():
            category_key_idx = i
            break

    if category_key_idx is None:
        return None

    for i in range(category_key_idx + 1, len(generated_tokens)):
        if generated_tokens[i].strip().strip('"') in ["Major", "Control"]:
            return i

    return None


def get_label_sequence_prob(
    scores: Tuple, token_start_idx: int, label_token_ids: List[int]
) -> float:
    """
    Length-normalised sequence probability:
        score = exp( mean_k [ log P(t_k | context) ] )

    Corrects length bias when comparing labels of different token counts
    e.g. 'Control' (1 token) vs 'Major Depressive Disorder' (4 tokens).
    """
    log_probs = []
    for offset, token_id in enumerate(label_token_ids):
        idx = token_start_idx + offset
        if idx >= len(scores):
            break
        lp = torch.log_softmax(scores[idx][0], dim=-1)[token_id].item()
        log_probs.append(lp)
    return float(np.exp(np.mean(log_probs))) if log_probs else 0.0


def compute_binary_probs(
    scores: Tuple,
    category_token_idx: int,
    mdd_token_ids: List[int],
    control_token_ids: List[int],
) -> Dict[str, float]:
    """
    Returns normalised P(MDD) and P(Control) plus the first-token
    decision margin (the actual fork point between the two labels).
    """
    p_mdd = get_label_sequence_prob(scores, category_token_idx, mdd_token_ids)
    p_control = get_label_sequence_prob(scores, category_token_idx, control_token_ids)
    total = p_mdd + p_control + 1e-12

    first_probs = torch.softmax(scores[category_token_idx][0], dim=-1)
    p_mdd_first = first_probs[mdd_token_ids[0]].item()
    p_ctrl_first = first_probs[control_token_ids[0]].item()

    return {
        "p_mdd_norm": p_mdd / total,
        "p_control_norm": p_control / total,
        "p_mdd_seq": p_mdd,
        "p_control_seq": p_control,
        "decision_margin": p_mdd_first - p_ctrl_first,  # + → leans MDD
        "pred": "Major Depressive Disorder"
        if p_mdd > p_control
        else "Control (no disorder detected)",
    }


class DataLoader:
    @staticmethod
    def load_text_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def get_mri_content(
        txt_filename: str, mri_base_path: str, include_images: bool = True
    ) -> List[Dict[str, Any]]:
        """Load MRI text parcellation data and/or brain visualisation images."""
        patient_id = os.path.splitext(txt_filename)[0]
        sub_folder = f"sub-{int(patient_id):04d}"
        subject_path = os.path.join(mri_base_path, sub_folder)

        if not os.path.exists(subject_path):
            return [{"type": "text", "text": f"No MRI data found for {sub_folder}"}]

        items = []
        for session in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path):
                continue
            items.append({"type": "text", "text": f"\n=== {session} ==="})
            for fname in sorted(os.listdir(session_path)):
                fpath = os.path.join(session_path, fname)
                if fname.endswith(".txt"):
                    items.append(
                        {
                            "type": "text",
                            "text": f"\n{fname}:\n{DataLoader.load_text_file(fpath)}",
                        }
                    )
                elif fname.endswith(".png") and include_images:
                    items.append({"type": "image", "image": Image.open(fpath)})
                    items.append({"type": "text", "text": f"[Image: {fname}]"})

        return items if items else [{"type": "text", "text": "No MRI data found"}]


class MinistralHandler:
    def __init__(self, config: InferenceConfig):
        self.config = config
        try:
            from transformers import (
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )
        except ImportError:
            raise ImportError(
                "MistralCommonBackend not found. Use a newer transformers environment."
            )

        print(f"Loading {config.model_name} ...")
        self.processor = MistralCommonBackend.from_pretrained(config.model_name)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            config.model_name, device_map="auto"
        )

        tok = getattr(
            self.processor, "tokenizer", getattr(self.processor, "text_tokenizer", None)
        )
        if tok is None:
            tok = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer = tok

        def encode_label(label: str) -> List[int]:
            dummy = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": label}]}],
                return_tensors="pt",
                return_dict=True,
            )
            ids = dummy["input_ids"][0].tolist()
            for start in range(len(ids)):
                decoded = self.tokenizer.decode(ids[start:])
                if label in decoded:
                    return ids[start:]
            return ids

        self.mdd_token_ids = encode_label("Major Depressive Disorder")
        self.control_token_ids = encode_label("Control")
        print(
            f"[DEBUG] MDD token ids:     {self.mdd_token_ids} "
            f"→ '{self.tokenizer.decode(self.mdd_token_ids)}'"
        )
        print(
            f"[DEBUG] Control token ids: {self.control_token_ids} "
            f"→ '{self.tokenizer.decode(self.control_token_ids)}'"
        )

    def _build_prompt(self, text: str, has_mri_data: bool = False) -> str:
        if has_mri_data:
            preamble = (
                "You are given patient clinical information and their MRI data "
                "(brain parcellation volume, visualization of brain regions)."
            )
        else:
            preamble = "You are given patient clinical information."

        prompt = preamble + "\n"
        prompt += """Classify the patient into one of the following categories:
- Major Depressive Disorder
- Control (no disorder detected)

Return your answer as a JSON object with two fields:
- "category": the chosen category (exactly one of the two above)
- "explanation": a short reasoning for the choice

"""
        prompt += f"Patient data:\n{text}\n\n"
        return prompt

    def _build_inputs(
        self,
        text: str,
        mri_content: Optional[List[Dict]],
        force_mri_preamble: bool = False,
    ) -> Dict:
        """Build tokenized inputs for Mistral3.

        force_mri_preamble=True uses the MRI preamble even when mri_content is
        None — isolates the effect of the preamble wording from actual MRI data
        (ablation mode: tabular_mri_preamble).
        """
        has_mri = bool(mri_content) or force_mri_preamble
        prompt_text = self._build_prompt(text, has_mri_data=has_mri)
        user_content = [{"type": "text", "text": prompt_text}]

        if mri_content:
            for item in mri_content:
                if item["type"] == "text":
                    user_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_data_url(item["image"])
                            },
                        }
                    )

        messages = [{"role": "user", "content": user_content}]
        tokenized = self.processor.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        )

        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.to(self.model.device)
        if "pixel_values" in tokenized:
            tokenized["pixel_values"] = tokenized["pixel_values"].to(
                dtype=torch.bfloat16
            )

        return tokenized

    @staticmethod
    def _image_to_data_url(image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _get_image_sizes(tokenized: Dict) -> Optional[List[Tuple]]:
        if "pixel_values" not in tokenized:
            return None
        h, w = tokenized["pixel_values"].shape[-2:]
        n_imgs = tokenized["pixel_values"].shape[0]
        return [(h, w)] * n_imgs

    def classify(
        self,
        text: str,
        mri_content: Optional[List[Dict]] = None,
        force_mri_preamble: bool = False,
    ) -> str:
        tokenized = self._build_inputs(text, mri_content, force_mri_preamble)
        image_sizes = self._get_image_sizes(tokenized)
        inp_len = tokenized["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **tokenized,
                image_sizes=image_sizes,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )[0]

        return self.processor.decode(output[inp_len:]).strip()

    def _generate_with_scores(
        self,
        text: str,
        mri_content: Optional[List[Dict]],
        force_mri_preamble: bool = False,
    ) -> Tuple[str, Dict]:
        tokenized = self._build_inputs(text, mri_content, force_mri_preamble)
        image_sizes = self._get_image_sizes(tokenized)
        inp_len = tokenized["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **tokenized,
                image_sizes=image_sizes,
                max_new_tokens=self.config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        generated_ids = generation.sequences[0][inp_len:]
        id_list = generated_ids.tolist()
        generated_tokens = [self.tokenizer.decode([t]) for t in id_list]

        cat_idx = find_category_token_idx(generated_tokens)

        if cat_idx is None:
            decoded = self.tokenizer.decode(id_list)
            pred = (
                "Major Depressive Disorder"
                if "Major" in decoded
                else "Control (no disorder detected)"
            )
            tqdm.write(f"  [WARN] cat_idx not found, falling back → {pred}")
            return pred, {
                "p_mdd_norm": 0.5,
                "p_control_norm": 0.5,
                "p_mdd_seq": 0.0,
                "p_control_seq": 0.0,
                "decision_margin": 0.0,
                "pred": pred,
            }

        probs = compute_binary_probs(
            generation.scores, cat_idx, self.mdd_token_ids, self.control_token_ids
        )
        return probs["pred"], probs

    def classify_contrastive(
        self,
        text: str,
        mri_content: Optional[List[Dict]],
        mode: str = "tabular_parcel_mri",
    ) -> Dict[str, Any]:
        """
        Compares two conditions depending on mode:

        tabular_parcel_mri   → baseline (no preamble, no MRI) vs full (preamble + MRI data)
        tabular_mri_preamble → baseline (no preamble, no MRI) vs full (preamble only, no MRI data)
        """
        if mode == "tabular_mri_preamble":
            variants = [
                ("baseline", None, False),  # plain tabular
                ("full", None, True),  # MRI preamble, no MRI data
            ]
        else:  # tabular_parcel_mri
            variants = [
                ("baseline", None, False),  # plain tabular
                ("full", mri_content, False),  # preamble + full MRI data
            ]

        results = {}
        for variant, data, force_preamble in variants:
            pred, probs = self._generate_with_scores(text, data, force_preamble)
            results[variant] = probs

        results["delta_p_mdd"] = (
            results["full"]["p_mdd_norm"] - results["baseline"]["p_mdd_norm"]
        )
        return results


class Qwen2_5VLHandler:
    def __init__(self, config: InferenceConfig):
        self.config = config
        print(f"Loading {config.model_name} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(config.model_name)

        tok = self.processor.tokenizer
        self.mdd_token_ids = tok.encode(
            "Major Depressive Disorder", add_special_tokens=False
        )
        self.control_token_ids = tok.encode("Control", add_special_tokens=False)

    def _build_prompt(self, text: str, has_mri_data: bool = False) -> str:
        if has_mri_data:
            preamble = (
                "You are given patient clinical information and their MRI data "
                "(brain parcellation volume, visualization of brain regions)."
            )
        else:
            preamble = "You are given patient clinical information."

        prompt = preamble + "\n"
        prompt += """Classify the patient into one of the following categories:
- Major Depressive Disorder
- Control (no disorder detected)

Return your answer as a JSON object with two fields:
- "category": the chosen category (exactly one of the two above)
- "explanation": a short reasoning for the choice

"""
        prompt += f"Patient data:\n{text}\n\n"
        return prompt

    def _build_messages(
        self,
        text: str,
        mri_content: Optional[List[Dict]],
        force_mri_preamble: bool = False,
    ) -> List[Dict]:
        """Build the message list for generation.

        force_mri_preamble=True uses the MRI preamble even when mri_content is
        None — isolates the effect of the preamble wording from actual MRI data
        (ablation mode: tabular_mri_preamble).
        """
        has_mri = bool(mri_content) or force_mri_preamble
        prompt_text = self._build_prompt(text, has_mri_data=has_mri)
        user_content = [{"type": "text", "text": prompt_text}]

        if mri_content:
            user_content += mri_content

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": user_content},
        ]

    def _generate(self, messages: List[Dict]) -> str:
        tmpl = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        img_in, vid_in = process_vision_info(messages)
        inputs = self.processor(
            text=[tmpl], images=img_in, videos=vid_in, padding=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        inp_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        return self.processor.decode(gen[0][inp_len:], skip_special_tokens=True).strip()

    def _generate_with_scores(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Generate with output_scores=True → return (pred_label, prob_dict)."""
        tmpl = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        img_in, vid_in = process_vision_info(messages)
        inputs = self.processor(
            text=[tmpl], images=img_in, videos=vid_in, padding=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        inp_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,  # must be greedy for deterministic probs
            )

        generated_ids = generation.sequences[0][inp_len:]
        generated_tokens = [self.processor.tokenizer.decode(t) for t in generated_ids]

        cat_idx = find_category_token_idx(generated_tokens)
        if cat_idx is None:
            decoded = self.processor.decode(generated_ids, skip_special_tokens=True)
            pred = (
                "Major Depressive Disorder"
                if "Major" in decoded
                else "Control (no disorder detected)"
            )
            return pred, {
                "p_mdd_norm": 0.5,
                "p_control_norm": 0.5,
                "p_mdd_seq": 0.0,
                "p_control_seq": 0.0,
                "decision_margin": 0.0,
                "pred": pred,
            }

        probs = compute_binary_probs(
            generation.scores, cat_idx, self.mdd_token_ids, self.control_token_ids
        )
        return probs["pred"], probs

    def classify(
        self,
        text: str,
        mri_content: Optional[List[Dict]] = None,
        force_mri_preamble: bool = False,
    ) -> str:
        return self._generate(
            self._build_messages(text, mri_content, force_mri_preamble)
        )

    def classify_contrastive(
        self,
        text: str,
        mri_content: Optional[List[Dict]],
        mode: str = "tabular_parcel_mri",
    ) -> Dict[str, Any]:
        """
        Compares two conditions depending on mode:

        tabular_parcel_mri   → baseline (no preamble, no MRI) vs full (preamble + MRI data)
        tabular_mri_preamble → baseline (no preamble, no MRI) vs full (preamble only, no MRI data)
        """
        if mode == "tabular_mri_preamble":
            variants = [
                ("baseline", None, False),  # plain tabular
                ("full", None, True),  # MRI preamble, no MRI data
            ]
        else:  # tabular_parcel_mri
            variants = [
                ("baseline", None, False),  # plain tabular
                ("full", mri_content, False),  # preamble + full MRI data
            ]

        results = {}
        for variant, data, force_preamble in variants:
            msgs = self._build_messages(text, data, force_preamble)
            pred, probs = self._generate_with_scores(msgs)
            results[variant] = probs

        results["delta_p_mdd"] = (
            results["full"]["p_mdd_norm"] - results["baseline"]["p_mdd_norm"]
        )
        return results


def create_handler(config: InferenceConfig):
    """Instantiate the correct handler based on model name."""
    n = config.model_name.lower()
    if "ministral" in n or ("mistral" in n and "2512" in n):
        return MinistralHandler(config)
    elif "qwen2.5-vl" in n or "qwen2_5_vl" in n:
        return Qwen2_5VLHandler(config)
    else:
        raise ValueError(
            f"Unsupported model: {config.model_name}. "
            "Supported: Qwen2.5-VL, Ministral/Mistral3"
        )


class InferencePipeline:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.handler = create_handler(config)
        self.data_loader = DataLoader()

    def _iter_patients(self) -> Tuple[str, str]:
        """Iterate over all MDD patients. Yields (filename, txt_filepath)."""
        for filename in sorted(os.listdir(self.config.txt_mdd_path)):
            if filename.lower().endswith(".txt"):
                yield filename, os.path.join(self.config.txt_mdd_path, filename)

    def _load_patient(
        self, filename: str, txt_filepath: str
    ) -> Tuple[str, Optional[List[Dict]]]:
        text = self.data_loader.load_text_file(txt_filepath)
        mri = None
        if self.config.mri_mdd_path:
            mri = self.data_loader.get_mri_content(
                filename, self.config.mri_mdd_path, include_images=True
            )
        return text, mri

    def _summarize_mri(self, mri_content: List[Dict]) -> str:
        parts = []
        for item in mri_content:
            if item["type"] == "text":
                parts.append(item["text"])
            elif item["type"] == "image":
                parts.append("[Image data included]")
        return "\n".join(parts)

    def run(self, mode: str = "tabular"):
        """
        Modes
        -----
        tabular
            Plain clinical text, no MRI preamble, no MRI data.

        tabular_mri_preamble
            Plain clinical text + MRI preamble wording, but NO actual MRI
            data appended. Isolates whether the preamble phrase alone shifts
            the model's predictions relative to 'tabular'.

        tabular_parcel_mri
            Clinical text + MRI preamble + full MRI content (parcellation
            tables + brain images).
        """
        force_preamble = mode == "tabular_mri_preamble"
        use_mri = mode == "tabular_parcel_mri"

        with open(self.config.output_file, "w", encoding="utf-8") as out_f:
            for filename, txt_filepath in tqdm(list(self._iter_patients())):
                text, mri = self._load_patient(filename, txt_filepath)
                mri_input = mri if use_mri else None
                output = self.handler.classify(
                    text, mri_input, force_mri_preamble=force_preamble
                )
                record = {
                    "filename": filename,
                    "true_label": "mdd",
                    "input": text,
                    "output": output,
                    "timestamp": datetime.now().isoformat(),
                }
                if mri and use_mri:
                    record["mri_data_summary"] = self._summarize_mri(mri)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{filename}] → {output[:80]}")

    def run_contrastive(self, mode: str = "tabular_parcel_mri"):
        """
        For every true MDD patient, run both conditions and record ALL outcomes.

        The contrastive pair depends on mode:
          tabular_parcel_mri   → baseline (plain) vs full (preamble + MRI data)
          tabular_mri_preamble → baseline (plain) vs full (preamble only, no MRI data)

        Case types (true label is always MDD):
          Type_A_both_correct  — baseline=MDD, full=MDD  (both right)
          Type_B_recovered     — baseline=Ctrl, full=MDD  (condition rescued)
          Type_C_regressed     — baseline=MDD, full=Ctrl  (condition hurt)
          Type_D_both_wrong    — baseline=Ctrl, full=Ctrl  (both wrong)
        """
        out_path = self.config.output_file.replace(".jsonl", "_contrastive.jsonl")
        print(f"Contrastive results → {out_path}")
        n_total = 0
        counts = {
            "Type_A_both_correct": 0,
            "Type_B_recovered": 0,
            "Type_C_regressed": 0,
            "Type_D_both_wrong": 0,
        }

        with open(out_path, "w", encoding="utf-8") as out_f:
            for filename, txt_filepath in tqdm(list(self._iter_patients())):
                text, mri = self._load_patient(filename, txt_filepath)
                contrastive = self.handler.classify_contrastive(text, mri, mode=mode)
                n_total += 1

                base_is_mdd = "Major" in contrastive["baseline"]["pred"]
                full_is_mdd = "Major" in contrastive["full"]["pred"]

                if base_is_mdd and full_is_mdd:
                    case_type = "Type_A_both_correct"
                elif not base_is_mdd and full_is_mdd:
                    case_type = "Type_B_recovered"
                elif base_is_mdd and not full_is_mdd:
                    case_type = "Type_C_regressed"
                else:
                    case_type = "Type_D_both_wrong"

                counts[case_type] += 1

                record = {
                    "filename": filename,
                    "true_label": "mdd",
                    "case_type": case_type,
                    "baseline": contrastive["baseline"],
                    "full": contrastive["full"],
                    "delta_p_mdd": contrastive["delta_p_mdd"],
                    "timestamp": datetime.now().isoformat(),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"✓ [{filename}] {case_type} | "
                    f"base p_mdd={contrastive['baseline']['p_mdd_norm']:.3e} → "
                    f"full p_mdd={contrastive['full']['p_mdd_norm']:.3e} "
                    f"Δ={contrastive['delta_p_mdd']:+.3e}"
                )

        print(f"\n{'=' * 50}")
        print(f"Contrastive summary ({n_total} MDD patients):")
        print(f"  Type A (both correct): {counts['Type_A_both_correct']:>4}")
        print(f"  Type B (recovered):    {counts['Type_B_recovered']:>4}")
        print(f"  Type C (regressed):    {counts['Type_C_regressed']:>4}")
        print(f"  Type D (both wrong):   {counts['Type_D_both_wrong']:>4}")
        print(f"{'=' * 50}")
        print(f"Saved to {out_path}")
        return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen2.5-VL / Ministral — MDD inference & ablation"
    )
    p.add_argument(
        "--txt_mdd_base_path",
        type=str,
        required=True,
        help="Base path for MDD text files, e.g. .../txt_mdd_split",
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tabular", "tabular_mri_preamble", "tabular_parcel_mri"],
    )
    p.add_argument(
        "--mri_mdd_base_path",
        type=str,
        default=None,
        help="Base path for MDD MRI data (required only for tabular_parcel_mri)",
    )
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--splits", type=str, nargs="+", default=["train", "dev", "test"])
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--run_contrastive", action="store_true")
    return p.parse_args()


def validate_args(args):
    # Only tabular_parcel_mri physically loads MRI files — other modes don't need the path
    if args.mode == "tabular_parcel_mri" and not args.mri_mdd_base_path:
        raise ValueError("tabular_parcel_mri requires --mri_mdd_base_path")
    if not os.path.exists(args.txt_mdd_base_path):
        raise FileNotFoundError(args.txt_mdd_base_path)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


if __name__ == "__main__":
    args = validate_args(parse_args())
    set_seed(args.seed)

    mb = args.model_name.split("/")[-1].replace("-", "_")

    first_split = args.splits[0]
    config = InferenceConfig(
        txt_mdd_path=os.path.join(args.txt_mdd_base_path, first_split),
        mri_mdd_path=os.path.join(args.mri_mdd_base_path, first_split)
        if args.mri_mdd_base_path
        else None,
        output_file=os.path.join(
            args.output_dir, f"results_{mb}_{args.mode}_{first_split}.jsonl"
        ),
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
    )

    print("=" * 60)
    print(f"Model      : {config.model_name}")
    print(f"Mode       : {args.mode}")
    print(f"Splits     : {args.splits}")
    print(f"Contrastive: {args.run_contrastive}")
    print("=" * 60)

    pipeline = InferencePipeline(config)

    for split in args.splits:
        print(f"\n{'=' * 60}\nSplit: {split}\n{'=' * 60}")

        pipeline.config.txt_mdd_path = os.path.join(args.txt_mdd_base_path, split)
        pipeline.config.mri_mdd_path = (
            os.path.join(args.mri_mdd_base_path, split)
            if args.mri_mdd_base_path
            else None
        )
        pipeline.config.output_file = os.path.join(
            args.output_dir, f"results_{mb}_{args.mode}_{split}.jsonl"
        )

        if not os.path.exists(pipeline.config.txt_mdd_path):
            print(f"  Skipping — path not found: {pipeline.config.txt_mdd_path}")
            continue

        if args.run_contrastive:
            pipeline.run_contrastive(mode=args.mode)
        else:
            pipeline.run(mode=args.mode)

    print(f"\nAll splits done! Results in: {args.output_dir}")
