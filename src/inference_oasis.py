"""
Modular inference pipeline for patient cognitive-decline classification on the OASIS dataset.
Supports multimodal modes combining tabular text, MRI parcel reports, and MRI images.

Modes:
  - tabular:           text data only
  - tabular_parcel:    text + MRI parcel text
  - tabular_mri:       text + MRI images (no parcel text)
  - tabular_parcel_mri: text + parcel text + MRI images
  - parcel_mri:        parcel text + MRI images only

Usage:
    python inference_oasis.py \
        --txt_path      /data/OASIS-MRI \
        --mri_base_path /data/OASIS-MRI \
        --model_name    Qwen/Qwen2.5-VL-3B-Instruct \
        --mode          tabular_mri
"""

import os
import io
import json
import torch
import base64
import argparse
import numpy as np
import torchvision.transforms as T
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode

# HuggingFace Imports
from transformers import (
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    Glm4vForConditionalGeneration,
    set_seed
)


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline"""
    txt_path: str
    mri_base_path: Optional[str] = None
    output_file: str = "results.jsonl"
    model_name: str = "google/gemma-3-27b-it"
    max_new_tokens: int = 4096
    do_sample: bool = False
    categories: List[str] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = [
                "Major Depressive Disorder",
                "Control (no disorder detected)"
            ]


class DataLoader:
    """Handles loading of text and MRI data"""

    @staticmethod
    def load_text_file(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def _iter_atlas_reports(subject_path: str):
        """
        Yield (anat_folder_name, atlas_reports_path) for each anat sub-folder.
        Structure: subject_path/anat*/NIFTI/atlas_reports/
        """
        for anat_folder in sorted(os.listdir(subject_path)):
            anat_path = os.path.join(subject_path, anat_folder)
            if not os.path.isdir(anat_path):
                continue
            atlas_path = os.path.join(anat_path, "NIFTI", "atlas_reports")
            if os.path.isdir(atlas_path):
                yield anat_folder, atlas_path

    @staticmethod
    def get_patient_text(subject_path: str) -> str:
        """
        Collect all region_descriptions.txt files found under
        subject_path/anat*/NIFTI/atlas_reports/ and return them as one string.
        """
        parts = []
        for anat_folder, atlas_path in DataLoader._iter_atlas_reports(subject_path):
            txt_file = os.path.join(atlas_path, "region_descriptions.txt")
            if os.path.isfile(txt_file):
                content = DataLoader.load_text_file(txt_file)
                parts.append(f"=== {anat_folder} ===\n{content}")
        return "\n\n".join(parts) if parts else "No region descriptions found"

    @staticmethod
    def get_mri_content(subject_path: str,
                        include_images: bool = False) -> List[Dict[str, Any]]:
        """Extract MRI data (text and/or images) for a subject folder."""
        if not os.path.exists(subject_path):
            return [{"type": "text", "text": f"No MRI data found at {subject_path}"}]

        content_items = []
        for anat_folder, atlas_path in DataLoader._iter_atlas_reports(subject_path):
            content_items.append({"type": "text", "text": f"\n=== {anat_folder} ==="})
            for file in sorted(os.listdir(atlas_path)):
                file_path = os.path.join(atlas_path, file)
                if file.endswith('.txt'):
                    content = DataLoader.load_text_file(file_path)
                    content_items.append({"type": "text", "text": f"\n{file}:\n{content}"})
                elif file.endswith('.png') and include_images:
                    image = Image.open(file_path)
                    content_items.append({"type": "image", "image": image})
                    content_items.append({"type": "text", "text": f"[Image: {file}]"})

        return content_items if content_items else [{"type": "text", "text": "No MRI data found"}]

    @staticmethod
    def get_mri_images_only(subject_path: str) -> List[Dict[str, Any]]:
        """Get MRI images only (no text data)."""
        if not os.path.exists(subject_path):
            return [{"type": "text", "text": f"No MRI data found at {subject_path}"}]

        content_items = []
        for anat_folder, atlas_path in DataLoader._iter_atlas_reports(subject_path):
            content_items.append({"type": "text", "text": f"\n=== {anat_folder} ==="})
            for file in sorted(os.listdir(atlas_path)):
                if file.endswith('.png'):
                    image = Image.open(os.path.join(atlas_path, file))
                    content_items.append({"type": "image", "image": image})
                    content_items.append({"type": "text", "text": f"[Image: {file}]"})

        return content_items if content_items else [{"type": "text", "text": "No MRI images found"}]

    @staticmethod
    def get_mri_text_only(subject_path: str) -> str:
        """Get MRI data as text only (no images)."""
        content_items = DataLoader.get_mri_content(subject_path, include_images=False)
        return "\n".join(i["text"] for i in content_items if i["type"] == "text")


class ModelHandler(ABC):
    """Abstract base class for model handlers"""
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def classify(self, text: str, mri_content: Any) -> str:
        pass

    def _build_prompt(self, text: str, mri_data: Optional[str] = None,
                      include_patient_data: bool = True) -> str:
        """Build classification prompt"""
        prompt_parts = ["You are given patient"]
        if include_patient_data:
            #prompt_parts.append("clinical information")
            #prompt_parts.append("and their MRI data (brain parcellation volume, visualization of brain regions)")
            #prompt_parts.append("clinical information and The weather is sunny today")
            #prompt_parts.append("clinical information and fMRI data")
            prompt_parts.append("clinical information and Neuroimaging data is provided.")
        if mri_data:
            prompt_parts.append("and their MRI data (brain parcellation volume, visualization of brain regions)")

        prompt = " ".join(prompt_parts) + ".\n"
        prompt += """Classify the patient into one of the following categories:
- Cognitive Normal
- Cognitive Decline

Return your answer as a JSON object with two fields:
- "category": the chosen category (exactly one of the two above)
- "explanation": a short reasoning for the choice

"""
        if include_patient_data:
            prompt += f"Patient data:\n{text}\n\n"

        if mri_data and isinstance(mri_data, str) and mri_data != "multimodal":
            prompt += "Patient MRI parcellation data:\n"
            prompt += mri_data

        return prompt


class InternVLHandler(ModelHandler):
    """Handler for OpenGVLab/InternVL models."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def _load_model(self):
        print(f"Loading InternVL model: {self.config.model_name}...")
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=False
        )

    @staticmethod
    def build_transform(input_size):
        MEAN, STD = InternVLHandler.IMAGENET_MEAN, InternVLHandler.IMAGENET_STD
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = InternVLHandler.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def _process_image(self, image_obj, input_size=448, max_num=12):
        image = image_obj.convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(img) for img in images])
        return pixel_values

    def classify(self, text: str, mri_content: Any = None) -> str:
        pixel_values = None
        num_patches_list = None
        mri_text_part = ""

        if isinstance(mri_content, str):
            mri_text_part = mri_content
        elif isinstance(mri_content, list):
            pixel_values_list = []
            num_patches_list = []
            image_idx = 1
            for item in mri_content:
                if item["type"] == "text":
                    mri_text_part += item["text"] + "\n"
                elif item["type"] == "image":
                    mri_text_part += f"Image-{image_idx}: <image>\n"
                    pv = self._process_image(item["image"])
                    pixel_values_list.append(pv)
                    num_patches_list.append(pv.size(0))
                    image_idx += 1
            if pixel_values_list:
                pixel_values = torch.cat(pixel_values_list, dim=0).to(self.model.device, dtype=torch.bfloat16)

        if pixel_values is not None:
            question = self._build_prompt(text, "multimodal", include_patient_data=bool(text)) + "\n" + mri_text_part
        else:
            mri_arg = mri_text_part if mri_text_part.strip() else None
            question = self._build_prompt(text, mri_arg, include_patient_data=bool(text))

        generation_config = dict(max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample)
        response, _ = self.model.chat(
            self.tokenizer, pixel_values, question, generation_config,
            num_patches_list=num_patches_list, history=None, return_history=True
        )
        return response.strip()


class MinistralHandler(ModelHandler):
    """Handler for Mistral 3 (Ministral) models."""
    def _load_model(self):
        try:
            from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        except ImportError:
            raise ImportError(
                "MistralCommonBackend not found. "
                "You are likely running in the 'InternVL' environment. "
                "To use Ministral, switch to a newer environment with transformers."
            )
        print(f"Loading Ministral model: {self.config.model_name}...")
        self.processor = MistralCommonBackend.from_pretrained(self.config.model_name)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.config.model_name, device_map="auto"
        )

    @staticmethod
    def _image_to_data_url(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def classify(self, text: str, mri_content: Any = None) -> str:
        prompt_text = self._build_prompt(
            text,
            "multimodal" if isinstance(mri_content, list) else mri_content,
            include_patient_data=bool(text)
        )
        user_content_list = [{"type": "text", "text": prompt_text}]
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    user_content_list.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    data_url = self._image_to_data_url(item["image"])
                    user_content_list.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = [{"role": "user", "content": user_content_list}]
        tokenized = self.processor.apply_chat_template(messages, return_tensors="pt", return_dict=True)

        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.to(self.model.device)
        if "pixel_values" in tokenized:
            tokenized["pixel_values"] = tokenized["pixel_values"].to(dtype=torch.bfloat16, device=self.model.device)

        image_sizes = None
        if "pixel_values" in tokenized:
            h, w = tokenized["pixel_values"].shape[-2:]
            num_images = tokenized["pixel_values"].shape[0]
            image_sizes = [(h, w) for _ in range(num_images)]

        with torch.inference_mode():
            output = self.model.generate(
                **tokenized, image_sizes=image_sizes,
                max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )[0]
        return self.processor.decode(output[len(tokenized["input_ids"][0]):]).strip()


class PixtralHandler(ModelHandler):
    """Handler for Pixtral (Mistral Vision) models"""
    def _load_model(self):
        print(f"Loading Pixtral model: {self.config.model_name}...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:
        prompt_text = self._build_prompt(
            text,
            "multimodal" if isinstance(mri_content, list) else mri_content,
            include_patient_data=bool(text)
        )
        conversation_content = [{"type": "text", "text": prompt_text}]
        images_list = []
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    conversation_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    conversation_content.append({"type": "image"})
                    images_list.append(item["image"])

        conversation = [{"role": "user", "content": conversation_content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        if images_list:
            inputs = self.processor(text=prompt, images=images_list, return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        return self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()


class GemmaHandler(ModelHandler):
    """Handler for Gemma and MedGemma models"""
    def _load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text)}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text, mri_data)}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(self, text: str, mri_content_items: List[Dict]) -> str:
        user_content = [{"type": "text", "text": self._build_prompt(text, "multimodal", include_patient_data=bool(text))}]
        user_content.extend(mri_content_items)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        return self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()


class Qwen2VLHandler(ModelHandler):
    """Handler for Qwen 2 VL models"""
    def _load_model(self):
        from qwen_vl_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_name, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text)}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text, mri_data)}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(self, text: str, mri_content_items: List[Dict]) -> str:
        user_content = [{"type": "text", "text": self._build_prompt(text, "multimodal", include_patient_data=bool(text))}]
        user_content.extend(mri_content_items)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        return self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()


class Qwen2_5VLHandler(ModelHandler):
    """Handler for Qwen 2.5 VL models"""
    def _load_model(self):
        from qwen_vl_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text)}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text, mri_data)}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(self, text: str, mri_content_items: List[Dict]) -> str:
        user_content = [{"type": "text", "text": self._build_prompt(text, "multimodal", include_patient_data=bool(text))}]
        user_content.extend(mri_content_items)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        return self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()


class Qwen2_5Handler(ModelHandler):
    """Handler for Qwen 2.5 text-only models"""
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoTokenizer.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            raise ValueError("Qwen2.5 text model does not support multimodal input.")
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text)}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text, mri_data)}]},
        ]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        # Flatten content dicts to plain strings for text-only tokenizer
        formatted = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                if all(isinstance(x, str) for x in content):
                    content = " ".join(content)
                elif all(isinstance(x, dict) for x in content):
                    content = " ".join(x['text'] for x in content if 'text' in x)
            formatted.append({"role": msg["role"], "content": content})

        text = self.processor.apply_chat_template(formatted, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        return self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()


class Qwen3VLHandler(ModelHandler):
    """Handler for Qwen 3 VL models"""
    def _load_model(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text)}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text, mri_data)}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(self, text: str, mri_content_items: List[Dict]) -> str:
        user_content = [{"type": "text", "text": self._build_prompt(text, "multimodal", include_patient_data=bool(text))}]
        user_content.extend(mri_content_items)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        return self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()


class Qwen3VLMoeHandler(ModelHandler):
    """Handler for Qwen 3 VL MoE models"""
    def _load_model(self):
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.config.model_name, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text)}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(text, mri_data)}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(self, text: str, mri_content_items: List[Dict]) -> str:
        user_content = [{"type": "text", "text": self._build_prompt(text, "multimodal", include_patient_data=bool(text))}]
        user_content.extend(mri_content_items)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant in clinical psychiatry."}]},
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        return self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()


class LlavaOneVisionHandler(ModelHandler):
    """Handler for LLaVA-One-Vision models"""
    def _load_model(self):
        print(f"Loading LLaVA-One-Vision model: {self.config.model_name}...")
        from qwen_vl_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype="auto", device_map="auto",
            trust_remote_code=True, force_download=True
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name, trust_remote_code=True)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": self._build_prompt(text)}]}]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": self._build_prompt(text, mri_data)}]}]
        return self._generate(messages)

    def _classify_with_multimodal(self, text: str, mri_content_items: List[Dict]) -> str:
        user_content = [{"type": "text", "text": self._build_prompt(text, "multimodal", include_patient_data=bool(text))}]
        user_content.extend(mri_content_items)
        messages = [{"role": "user", "content": user_content}]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


class Glm4vHandler(ModelHandler):
    """Handler for GLM-4V models"""
    def _load_model(self):
        print(f"Loading GLM-4V model: {self.config.model_name}...")
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.config.model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.config.model_name, use_fast=True)

    def classify(self, text: str, mri_content: Any = None) -> str:
        prompt_text = self._build_prompt(
            text,
            "multimodal" if isinstance(mri_content, list) else mri_content,
            include_patient_data=bool(text)
        )
        user_content = []
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    user_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    user_content.append({"type": "image", "image": item["image"]})
            user_content.append({"type": "text", "text": prompt_text})
        else:
            user_content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": user_content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)
                if k in ("images", "pixel_values"):
                    inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
            )
        return self.processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()


class ModelFactory:
    """Factory for creating model handlers"""
    @staticmethod
    def create_handler(config: InferenceConfig) -> ModelHandler:
        model_name = config.model_name.lower()
        if "internvl" in model_name:
            return InternVLHandler(config)
        elif "ministral" in model_name or ("mistral" in model_name and "2512" in model_name):
            return MinistralHandler(config)
        elif "glm" in model_name:
            return Glm4vHandler(config)
        elif "llava-onevision" in model_name:
            return LlavaOneVisionHandler(config)
        elif "pixtral" in model_name:
            return PixtralHandler(config)
        elif "gemma" in model_name or "medgemma" in model_name:
            return GemmaHandler(config)
        elif "qwen2-vl" in model_name:
            return Qwen2VLHandler(config)
        elif "qwen2.5-vl" in model_name or "qwen2_5_vl" in model_name:
            return Qwen2_5VLHandler(config)
        elif "qwen3-vl" in model_name and ("moe" in model_name or "a3b" in model_name or "a22b" in model_name):
            return Qwen3VLMoeHandler(config)
        elif "qwen3-vl" in model_name:
            return Qwen3VLHandler(config)
        elif "qwen2.5" in model_name or "qwen2_5" in model_name:
            return Qwen2_5Handler(config)
        else:
            raise ValueError(f"Unsupported model: {config.model_name}")


class InferencePipeline:
    """Main inference pipeline"""
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.handler = ModelFactory.create_handler(config)
        self.data_loader = DataLoader()

    def _resolve_mri_subject_path(self, subject_folder: str, mri_base: str) -> Optional[str]:
        """
        Find the MRI folder for a given subject_folder name inside mri_base.

        Handles two cases:
          - txt folder name == mri folder name  (exact match)
          - txt folder is bare ID  e.g. 'OAS30089'
            mri folder has suffix  e.g. 'OAS30089_MR_d0001'  (prefix match)
        """
        # 1. Exact match
        exact = os.path.join(mri_base, subject_folder)
        if os.path.isdir(exact):
            return exact

        # 2. Prefix match: txt name 'OAS30089' -> mri name 'OAS30089_MR_d0001'
        #    Use the full subject_folder as prefix (works whether or not it has underscores)
        match = next(
            (os.path.join(mri_base, e)
             for e in sorted(os.listdir(mri_base))
             if e.startswith(subject_folder) and os.path.isdir(os.path.join(mri_base, e))),
            None
        )
        return match

    def run(self, mode: str = "tabular"):
        """
        Run inference pipeline.

        --txt_path  : flat folder of patient text files, one per subject.
                      File names are bare subject IDs, e.g. 'OAS30089' or 'OAS30089.txt'.
        --mri_base_path : root containing subject MRI folders, e.g. OAS30089_MR_d0001/.
                          Each subject folder must have the structure:
                              anat*/NIFTI/atlas_reports/region_descriptions.txt
                              anat*/NIFTI/atlas_reports/atlas_report.png
                          Defaults to txt_path if not provided.
        """
        base_path = self.config.txt_path
        mri_base  = self.config.mri_base_path or base_path
        needs_mri = mode in ("tabular_parcel", "tabular_mri", "tabular_parcel_mri", "parcel_mri")

        # ── Collect patient text files ────────────────────────────────────────
        all_entries = os.listdir(base_path)
        txt_files = sorted(
            e for e in all_entries
            if os.path.isfile(os.path.join(base_path, e))
        )

        print(f"[DEBUG] txt_path       : {base_path}")
        print(f"[DEBUG] mri_base_path  : {mri_base}")
        print(f"[DEBUG] mode           : {mode}")
        print(f"[DEBUG] total entries  : {len(all_entries)}  |  text files: {len(txt_files)}")
        if not txt_files:
            print("[ERROR] No files found in txt_path!")
            return
        print(f"[DEBUG] First files    : {txt_files[:5]}")
        print()

        with open(self.config.output_file, "w", encoding="utf-8") as out_f:
            for txt_file in tqdm(txt_files):
                txt_path = os.path.join(base_path, txt_file)
                # Subject ID is filename without extension, e.g. 'OAS30089'
                subject_id = os.path.splitext(txt_file)[0]

                # ── Load patient text from flat file ──────────────────────────
                patient_data = self.data_loader.load_text_file(txt_path)

                # ── Resolve MRI path ──────────────────────────────────────────
                if needs_mri:
                    mri_subject_path = self._resolve_mri_subject_path(subject_id, mri_base)
                    if mri_subject_path is None:
                        print(f"[WARN] No MRI folder found for '{subject_id}' in {mri_base} — skipping.")
                        continue
                else:
                    mri_subject_path = None

                # ── Classify ──────────────────────────────────────────────────
                mri_content = None
                mri_summary = None

                if mode == "tabular":
                    category = self.handler.classify(patient_data, None)


                elif mode == "tabular_parcel":
                    mri_text  = self.data_loader.get_mri_text_only(mri_subject_path)
                    mri_summary = mri_text
                    category  = self.handler.classify(patient_data, mri_text)

                elif mode == "tabular_mri":
                    mri_content = self.data_loader.get_mri_images_only(mri_subject_path)
                    mri_summary = self._summarize_mri_content(mri_content)
                    category    = self.handler.classify(patient_data, mri_content)

                elif mode == "tabular_parcel_mri":
                    mri_content = self.data_loader.get_mri_content(mri_subject_path, include_images=True)
                    mri_summary = self._summarize_mri_content(mri_content)
                    category    = self.handler.classify(patient_data, mri_content)

                elif mode == "parcel_mri":
                    mri_content = self.data_loader.get_mri_content(mri_subject_path, include_images=True)
                    mri_summary = self._summarize_mri_content(mri_content)
                    category    = self.handler.classify("", mri_content)

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # ── Write result ──────────────────────────────────────────────
                record = {
                    "subject_id":   subject_id,
                    "txt_path":     txt_path,
                    "input":        patient_data,
                    "output":       category,
                    "timestamp":    datetime.now().isoformat(),
                }
                if mri_summary:
                    record["mri_data_summary"] = mri_summary

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f"[{subject_id}] -> {category}")

    def _summarize_mri_content(self, mri_content: List[Dict]) -> str:
        parts = []
        for item in mri_content:
            if item["type"] == "text":
                parts.append(item["text"])
            elif item["type"] == "image":
                parts.append("[Image data included in processing]")
        return "\n".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular inference pipeline for patient classification using multimodal data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_oasis.py \\
    --txt_path /data/OASIS-MRI \\
    --mri_base_path /data/OASIS-MRI \\
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \\
    --mode tabular_mri
"""
    )
    parser.add_argument("--txt_path",      type=str, required=True,
                        help="Root directory containing subject folders (e.g. OAS30089_MR_d0001/)")
    parser.add_argument("--model_name",    type=str, required=True,
                        help="Model name or HuggingFace path")
    parser.add_argument("--mode",          type=str, required=True,
                        choices=["tabular", "tabular_parcel", "tabular_mri",
                                 "tabular_parcel_mri", "parcel_mri"],
                        help="Inference mode")
    parser.add_argument("--mri_base_path", type=str, default=None,
                        help="Root directory for MRI data (defaults to txt_path if omitted)")
    parser.add_argument("--output_file",   type=str, default=None,
                        help="Output JSONL file (auto-generated if omitted)")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--do_sample",     action="store_true")
    parser.add_argument("--seed",          type=int, default=666)
    return parser.parse_args()


def validate_args(args):
    if args.mode in ["tabular_parcel", "tabular_mri", "tabular_parcel_mri", "parcel_mri"]:
        if not args.mri_base_path:
            # Default to txt_path — print info so user knows
            print(f"[INFO] --mri_base_path not set; defaulting to --txt_path ({args.txt_path})")

    if not os.path.exists(args.txt_path):
        raise FileNotFoundError(f"txt_path does not exist: {args.txt_path}")
    if args.mri_base_path and not os.path.exists(args.mri_base_path):
        raise FileNotFoundError(f"mri_base_path does not exist: {args.mri_base_path}")

    if not args.output_file:
        model_base = args.model_name.split('/')[-1].replace('-', '_')
        txt_base   = "_".join(args.txt_path.rstrip('/').split('/')[-2:])
        mri_base   = "_".join(args.mri_base_path.rstrip('/').split('/')[-2:]) if args.mri_base_path else "same_as_txt"
        args.output_file = f"oasis_results_{model_base}_{txt_base}_{mri_base}_{args.mode}.jsonl"
        print(f"[INFO] Output file: {args.output_file}")

    return args


if __name__ == "__main__":
    args = parse_args()
    args = validate_args(args)

    set_seed(args.seed)

    config = InferenceConfig(
        txt_path       = args.txt_path,
        mri_base_path  = args.mri_base_path,
        output_file    = args.output_file,
        model_name     = args.model_name,
        max_new_tokens = args.max_new_tokens,
        do_sample      = args.do_sample,
    )

    print("=" * 60)
    print("INFERENCE CONFIGURATION")
    print("=" * 60)
    print(f"Model          : {config.model_name}")
    print(f"Mode           : {args.mode}")
    print(f"txt_path       : {config.txt_path}")
    print(f"mri_base_path  : {config.mri_base_path}")
    print(f"Output file    : {config.output_file}")
    print(f"Max new tokens : {config.max_new_tokens}")
    print(f"Do sample      : {config.do_sample}")
    print(f"Seed           : {args.seed}")
    print("=" * 60)
    print()

    print("Initializing model...")
    pipeline = InferencePipeline(config)
    print(f"Running inference in '{args.mode}' mode...")
    pipeline.run(mode=args.mode)

    print()
    print("=" * 60)
    print(f"Done! Results saved to: {config.output_file}")
    print("=" * 60)