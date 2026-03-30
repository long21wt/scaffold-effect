"""
Modular inference pipeline for patient MDD classification using multimodal data (tabular, parcel, MRI).
Supports InternVL, Ministral, Gemma, and other HuggingFace VLMs.

Modes:
  - tabular:           text data only
  - tabular_parcel:    text + MRI parcel text
  - tabular_mri:       text + MRI images (no parcel text)
  - tabular_parcel_mri: text + parcel text + MRI images
  - parcel_mri:        parcel text + MRI images only

Usage:
    # e.g., InternVL
    python inference.py \
        --txt_path      /path/to/txt \
        --mri_base_path /path/to/mri \
        --model_name    OpenGVLab/InternVL3_5-8B \
        --mode          tabular_parcel_mri
"""

import argparse
import base64
import io
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# HuggingFace Imports
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Glm4vForConditionalGeneration,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    set_seed,
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
                "Control (no disorder detected)",
            ]


class DataLoader:
    """Handles loading of text and MRI data"""

    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Load text file content"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def get_mri_content(
        txt_filename: str, mri_base_path: str, include_images: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract MRI data for a given text filename."""
        patient_id = os.path.splitext(txt_filename)[0]
        sub_folder = f"sub-{int(patient_id):04d}"
        subject_path = os.path.join(mri_base_path, sub_folder)

        if not os.path.exists(subject_path):
            return [{"type": "text", "text": f"No MRI data found for {sub_folder}"}]

        content_items = []
        for session_folder in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session_folder)
            if not os.path.isdir(session_path):
                continue
            content_items.append(
                {"type": "text", "text": f"\n=== {session_folder} ==="}
            )

            for file in sorted(os.listdir(session_path)):
                file_path = os.path.join(session_path, file)
                if file.endswith(".txt"):
                    content = DataLoader.load_text_file(file_path)
                    content_items.append(
                        {"type": "text", "text": f"\n{file}:\n{content}"}
                    )
                elif file.endswith(".png") and include_images:
                    image = Image.open(file_path)
                    content_items.append({"type": "image", "image": image})
                    content_items.append({"type": "text", "text": f"[Image: {file}]"})

        return (
            content_items
            if content_items
            else [{"type": "text", "text": "No MRI data found"}]
        )

    @staticmethod
    def get_mri_images_only(
        txt_filename: str, mri_base_path: str
    ) -> List[Dict[str, Any]]:
        """Get MRI images only (no text data)"""
        patient_id = os.path.splitext(txt_filename)[0]
        sub_folder = f"sub-{int(patient_id):04d}"
        subject_path = os.path.join(mri_base_path, sub_folder)

        if not os.path.exists(subject_path):
            return [{"type": "text", "text": f"No MRI data found for {sub_folder}"}]

        content_items = []
        for session_folder in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session_folder)
            if not os.path.isdir(session_path):
                continue
            content_items.append(
                {"type": "text", "text": f"\n=== {session_folder} ==="}
            )
            for file in sorted(os.listdir(session_path)):
                file_path = os.path.join(session_path, file)
                if file.endswith(".png"):
                    image = Image.open(file_path)
                    content_items.append({"type": "image", "image": image})
                    content_items.append({"type": "text", "text": f"[Image: {file}]"})

        return (
            content_items
            if content_items
            else [{"type": "text", "text": "No MRI images found"}]
        )

    @staticmethod
    def get_mri_text_only(txt_filename: str, mri_base_path: str) -> str:
        """Get MRI data as text only (no images)"""
        content_items = DataLoader.get_mri_content(
            txt_filename, mri_base_path, include_images=False
        )
        text_parts = [item["text"] for item in content_items if item["type"] == "text"]
        return "\n".join(text_parts) if text_parts else "No MRI text data found"


class ModelHandler(ABC):
    """Abstract base class for model handlers"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load model and processor"""
        pass

    @abstractmethod
    def classify(self, text: str, mri_content: Any) -> str:
        """Classify patient data"""
        pass

    def _build_prompt(
        self,
        text: str,
        mri_data: Optional[str] = None,
        include_patient_data: bool = True,
    ) -> str:
        """Build classification prompt"""
        prompt_parts = ["You are given patient"]
        if include_patient_data:
            # prompt_parts.append("clinical information")
            # prompt_parts.append("clinical information and their MRI data")
            # prompt_parts.append("clinical information and (brain parcellation volume, visualization of brain regions)")
            # prompt_parts.append("clinical information and their MRI data (brain parcellation volume, visualization of brain regions)")
            # prompt_parts.append("clinical information and their CT Scan data")
            # prompt_parts.append("clinical information and their Magnetic Resonance Imaging data")
            # prompt_parts.append("clinical information and their Magnetic Resonance Imaging data (brain parcellation volume, visualization of brain regions)")
            # prompt_parts.append("clinical information and fMRI data")
            # prompt_parts.append("clinical information and Neuroimaging data is provided.")
            prompt_parts.append("clinical information and The weather is sunny today")
        if mri_data:
            prompt_parts.append(
                "and their MRI data (brain parcellation volume, visualization of brain regions)"
            )

        prompt = " ".join(prompt_parts) + ".\n"
        #        prompt += """Classify the patient into one of the following categories:
        # - Major Depressive Disorder
        # - Control (no disorder detected)

        # Don't think, return your answer as a JSON object with one fields:
        # - "category": the chosen category (exactly one of the two above)

        # """
        prompt += """Classify the patient into one of the following categories:
- Major Depressive Disorder
- Control (no disorder detected)

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
    """
    Handler for OpenGVLab/InternVL models.
    """

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
            device_map="auto",
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, use_fast=False
        )

    # --- InternVL Helper Methods ---
    @staticmethod
    def build_transform(input_size):
        MEAN, STD = InternVLHandler.IMAGENET_MEAN, InternVLHandler.IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    @staticmethod
    def find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
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
    def dynamic_preprocess(
        image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = InternVLHandler.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
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
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _process_image(self, image_obj, input_size=448, max_num=12):
        image = image_obj.convert("RGB")
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
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
                pixel_values = torch.cat(pixel_values_list, dim=0)
                pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16)

        if pixel_values is not None:
            base_prompt = self._build_prompt(
                text, "multimodal", include_patient_data=bool(text)
            )
            question = base_prompt + "\n" + mri_text_part
        else:
            mri_arg = mri_text_part if mri_text_part.strip() else None
            question = self._build_prompt(
                text, mri_arg, include_patient_data=bool(text)
            )

        generation_config = dict(
            max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample
        )

        # Get the original pixel_values
        # original_pixel_values = pixel_values

        # Check the statistics of the original values
        # original_min = original_pixel_values.min()
        # original_max = original_pixel_values.max()
        # original_mean = original_pixel_values.mean()
        # original_std = original_pixel_values.std()

        # print("Original pixel_values statistics:")
        # print(f"  Shape: {original_pixel_values.shape}")
        # print(f"  Min: {original_min.item():.4f}")
        # print(f"  Max: {original_max.item():.4f}")
        # print(f"  Mean: {original_mean.item():.4f}")
        # print(f"  Std: {original_std.item():.4f}")

        # Generate Gaussian noise matching the original distribution
        # original_shape = original_pixel_values.shape
        # device = original_pixel_values.device
        # dtype = original_pixel_values.dtype

        # Generate Gaussian noise with matching statistics
        # gaussian_noise = torch.randn(original_shape, device=device, dtype=dtype) * original_std + original_mean
        # zeros = torch.zeros(original_shape, device=device, dtype=dtype)

        # Clip the noise to the original range
        # gaussian_noise = torch.clamp(gaussian_noise, min=original_min, max=original_max)

        # Verify the new noise statistics
        # print("\nClipped Gaussian noise statistics:")
        # print(f"  Min: {gaussian_noise.min().item():.4f}")
        # print(f"  Max: {gaussian_noise.max().item():.4f}")
        # print(f"  Mean: {gaussian_noise.mean().item():.4f}")
        # print(f"  Std: {gaussian_noise.std().item():.4f}")

        # Replace pixel_values with the clipped Gaussian noise
        # inputs['pixel_values'] = gaussian_noise
        # pixel_values = zeros

        response, _ = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
        return response.strip()


class MinistralHandler(ModelHandler):
    """
    Handler for Mistral 3 (Ministral) models.
    """

    def _load_model(self):
        # Lazy import to avoid crashing older environments used for InternVL
        try:
            from transformers import (
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )
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
        """Convert PIL image to base64 data URL"""
        buffered = io.BytesIO()
        # Save as PNG to ensure compatibility
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def classify(self, text: str, mri_content: Any = None) -> str:
        # 1. Build the prompt
        prompt_text = self._build_prompt(
            text,
            "multimodal" if isinstance(mri_content, list) else mri_content,
            include_patient_data=bool(text),
        )

        # 2. Construct Messages
        user_content_list = []
        user_content_list.append({"type": "text", "text": prompt_text})

        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    user_content_list.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    # FIX: Convert local PIL image to Data URI and use 'image_url' schema
                    # This satisfies strict schema validation in MistralCommonBackend
                    data_url = self._image_to_data_url(item["image"])
                    user_content_list.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )

        messages = [
            {
                "role": "user",
                "content": user_content_list,
            },
        ]

        # 3. Tokenize / Process
        # MistralCommonBackend processes the base64 images into pixel_values here
        tokenized = self.processor.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        )

        # 4. Move inputs to device
        # Loop through all keys (input_ids, pixel_values, attention_mask) and move to GPU
        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.to(self.model.device)

        # Ensure pixel_values are bfloat16
        if "pixel_values" in tokenized:
            tokenized["pixel_values"] = tokenized["pixel_values"].to(
                dtype=torch.bfloat16, device=self.model.device
            )

        # 5. Extract image sizes
        # The model requires image_sizes arg.
        # MistralCommonBackend usually normalizes images, so we extract dimensions from the tensor.
        image_sizes = None
        if "pixel_values" in tokenized:
            # pixel_values shape: (Batch, Channels, Height, Width) or (Batch, num_patches, ...)
            # We need a list of (H, W) tuples.
            # Depending on the backend processing, pixel_values might be 4D or 5D.
            # We take the spatial dims from the last two dimensions.
            h, w = tokenized["pixel_values"].shape[-2:]
            num_images = tokenized["pixel_values"].shape[0]
            image_sizes = [(h, w) for _ in range(num_images)]

        # Get the original pixel_values
        # original_pixel_values = tokenized['pixel_values']

        # Check the statistics of the original values
        # original_min = original_pixel_values.min()
        # original_max = original_pixel_values.max()
        # original_mean = original_pixel_values.mean()
        # original_std = original_pixel_values.std()

        # print("Original pixel_values statistics:")
        # print(f"  Shape: {original_pixel_values.shape}")
        # print(f"  Min: {original_min.item():.4f}")
        # print(f"  Max: {original_max.item():.4f}")
        # print(f"  Mean: {original_mean.item():.4f}")
        # print(f"  Std: {original_std.item():.4f}")

        # Generate Gaussian noise matching the original distribution
        # original_shape = original_pixel_values.shape
        # device = original_pixel_values.device
        # dtype = original_pixel_values.dtype

        # Generate Gaussian noise with matching statistics
        # gaussian_noise = torch.randn(original_shape, device=device, dtype=dtype) * original_std + original_mean
        # zeros = torch.zeros(original_shape, device=device, dtype=dtype)

        # Clip the noise to the original range
        # gaussian_noise = torch.clamp(gaussian_noise, min=original_min, max=original_max)

        # Verify the new noise statistics
        # print("\nClipped Gaussian noise statistics:")
        # print(f"  Min: {gaussian_noise.min().item():.4f}")
        # print(f"  Max: {gaussian_noise.max().item():.4f}")
        # print(f"  Mean: {gaussian_noise.mean().item():.4f}")
        # print(f"  Std: {gaussian_noise.std().item():.4f}")

        # Replace pixel_values with the clipped Gaussian noise
        # inputs['pixel_values'] = gaussian_noise
        # tokenized['pixel_values'] = zeros

        # print("\nPixel values replaced with clipped Gaussian noise!")

        # 6. Generate
        with torch.inference_mode():
            output = self.model.generate(
                **tokenized,
                image_sizes=image_sizes,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )[0]

        # 7. Decode
        decoded_output = self.processor.decode(output[len(tokenized["input_ids"][0]) :])
        return decoded_output.strip()


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
            include_patient_data=bool(text),
        )

        conversation_content = []
        images_list = []
        conversation_content.append({"type": "text", "text": prompt_text})
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    conversation_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    conversation_content.append({"type": "image"})
                    images_list.append(item["image"])
        conversation = [{"role": "user", "content": conversation_content}]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        if images_list:
            inputs = self.processor(
                text=prompt, images=images_list, return_tensors="pt"
            )
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")

        inputs = inputs.to(self.model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        generated_ids = output[0][input_len:]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()


class GemmaHandler(ModelHandler):
    """Handler for Gemma and MedGemma models"""

    def _load_model(self):
        model_id = self.config.model_name
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        prompt = self._build_prompt(text)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(
        self, text: str, mri_content_items: List[Dict]
    ) -> str:
        prompt_text = self._build_prompt(
            text, "multimodal", include_patient_data=bool(text)
        )
        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend(mri_content_items)
        messages = [
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
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        generation = generation[0][input_len:]
        return self.processor.decode(generation, skip_special_tokens=True).strip()


class Qwen2VLHandler(ModelHandler):
    """Handler for Qwen 2 VL models"""

    def _load_model(self):
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
        prompt = self._build_prompt(text)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(
        self, text: str, mri_content_items: List[Dict]
    ) -> str:
        prompt_text = self._build_prompt(
            text, "multimodal", include_patient_data=bool(text)
        )
        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend(mri_content_items)
        messages = [
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
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        generation = generation[0][input_len:]
        return self.processor.decode(generation, skip_special_tokens=True).strip()


class Qwen2_5VLHandler(ModelHandler):
    """Handler for Qwen 2.5 VL models"""

    def _load_model(self):
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        prompt = self._build_prompt(text)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(
        self, text: str, mri_content_items: List[Dict]
    ) -> str:
        prompt_text = self._build_prompt(
            text, "multimodal", include_patient_data=bool(text)
        )
        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend(mri_content_items)
        messages = [
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
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        generation = generation[0][input_len:]
        return self.processor.decode(generation, skip_special_tokens=True).strip()


class Qwen2_5Handler(ModelHandler):
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoTokenizer.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            raise ValueError("Qwen2.5 model does not support multimodal input.")
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        prompt = self._build_prompt(text)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        formatted_messages = []
        for msg in messages:
            content = msg["content"]

            # If content is a list of strings, join them
            if isinstance(content, list):
                # If it's a list of strings:
                if all(isinstance(x, str) for x in content):
                    content = " ".join(content)
                # If it's a multimodal list (dicts), extract only the text
                elif all(isinstance(x, dict) for x in content):
                    content = " ".join([x["text"] for x in content if "text" in x])

            formatted_messages.append({"role": msg["role"], "content": content})

        text = self.processor.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(
            self.model.device
        )

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        generation = generation[0][input_len:]
        return self.processor.decode(generation, skip_special_tokens=True).strip()


class Qwen3VLHandler(ModelHandler):
    """Handler for Qwen 3 VL models"""

    def _load_model(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
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
        prompt = self._build_prompt(text)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(
        self, text: str, mri_content_items: List[Dict]
    ) -> str:
        prompt_text = self._build_prompt(
            text, "multimodal", include_patient_data=bool(text)
        )
        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend(mri_content_items)
        messages = [
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
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        generation = generation[0][input_len:]
        return self.processor.decode(generation, skip_special_tokens=True).strip()


class Qwen3VLMoeHandler(ModelHandler):
    """Handler for Qwen 3 VL MoE models"""

    def _load_model(self):
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
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
        prompt = self._build_prompt(text)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(
        self, text: str, mri_content_items: List[Dict]
    ) -> str:
        prompt_text = self._build_prompt(
            text, "multimodal", include_patient_data=bool(text)
        )
        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend(mri_content_items)
        messages = [
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
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        generation = generation[0][input_len:]
        return self.processor.decode(generation, skip_special_tokens=True).strip()


class LlavaOneVisionHandler(ModelHandler):
    """Handler for LLaVA-One-Vision models"""

    def _load_model(self):
        print(f"Loading LLaVA-One-Vision model: {self.config.model_name}...")
        # Import process_vision_info locally as required by the snippet
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info

        # Load model with trust_remote_code=True and auto dtype as requested
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            force_download=True,
        )
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )

    def classify(self, text: str, mri_content: Any = None) -> str:
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        elif isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        else:
            return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        prompt = self._build_prompt(text)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(
        self, text: str, mri_content_items: List[Dict]
    ) -> str:
        prompt_text = self._build_prompt(
            text, "multimodal", include_patient_data=bool(text)
        )
        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend(mri_content_items)

        messages = [
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages)

    def _generate(self, messages: List[Dict]) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()


class Glm4vHandler(ModelHandler):
    """Handler for GLM-4V models"""

    def _load_model(self):
        print(f"Loading GLM-4V model: {self.config.model_name}...")
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, use_fast=True
        )

    def classify(self, text: str, mri_content: Any = None) -> str:
        prompt_text = self._build_prompt(
            text,
            "multimodal" if isinstance(mri_content, list) else mri_content,
            include_patient_data=bool(text),
        )

        user_content = []
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    user_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    user_content.append({"type": "image", "image": item["image"]})
            # Append the classification instruction/prompt
            user_content.append({"type": "text", "text": prompt_text})
        else:
            user_content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": user_content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # Move to device
                inputs[k] = v.to(self.model.device)

                # Force images/pixel_values to match model precision (bfloat16)
                # GLM-4V typically uses 'images'
                if k == "images" or k == "pixel_values":
                    inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        output_text = self.processor.decode(
            generated_ids[0][input_len:], skip_special_tokens=True
        )
        return output_text.strip()


class ModelFactory:
    """Factory for creating model handlers"""

    @staticmethod
    def create_handler(config: InferenceConfig) -> ModelHandler:
        """Create appropriate model handler based on model name"""
        model_name = config.model_name.lower()

        if "internvl" in model_name:
            return InternVLHandler(config)
        elif "ministral" in model_name or (
            "mistral" in model_name and "2512" in model_name
        ):
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
        elif "qwen3-vl" in model_name and (
            "moe" in model_name or "a3b" in model_name or "a22b" in model_name
        ):
            return Qwen3VLMoeHandler(config)
        elif "qwen3-vl" in model_name:
            return Qwen3VLHandler(config)
        elif "qwen2.5" in model_name or "qwen2_5" in model_name:
            return Qwen2_5Handler(config)
        else:
            raise ValueError(f"Unsupported model: {config.model_name}")


# --------------------
# Inference Pipeline
# --------------------


class InferencePipeline:
    """Main inference pipeline"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.handler = ModelFactory.create_handler(config)
        self.data_loader = DataLoader()

    def _get_mri_paths(self, txt_filename: str) -> Dict[str, Any]:
        """Get all MRI file paths for a patient"""
        if not self.config.mri_base_path:
            return None

        patient_id = os.path.splitext(txt_filename)[0]
        sub_folder = f"sub-{int(patient_id):04d}"
        subject_path = os.path.join(self.config.mri_base_path, sub_folder)

        if not os.path.exists(subject_path):
            return None

        mri_paths = {"subject_folder": subject_path, "sessions": {}}

        for session_folder in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session_folder)
            if not os.path.isdir(session_path):
                continue

            mri_paths["sessions"][session_folder] = {
                "session_path": session_path,
                "files": [],
            }

            for file in sorted(os.listdir(session_path)):
                file_path = os.path.join(session_path, file)
                mri_paths["sessions"][session_folder]["files"].append(file_path)

        return mri_paths

    def run(self, mode: str = "tabular"):
        """
        Run inference pipeline
        Args:
            mode: One of 'tabular', 'tabular_parcel', 'tabular_mri',
                  'tabular_parcel_mri', 'parcel_mri'
        """
        with open(self.config.output_file, "w", encoding="utf-8") as out_f:
            for filename in tqdm(os.listdir(self.config.txt_path)):
                if not filename.lower().endswith(".txt"):
                    continue

                file_path = os.path.join(self.config.txt_path, filename)
                patient_data = self.data_loader.load_text_file(file_path)

                # Determine what MRI data to include
                mri_content = None
                mri_summary = None
                mri_paths = None

                if mode == "tabular":
                    # Text data only
                    category = self.handler.classify(patient_data, None)
                elif mode == "tabular_parcel":
                    # Text data + MRI text data
                    if self.config.mri_base_path:
                        mri_content = self.data_loader.get_mri_text_only(
                            filename, self.config.mri_base_path
                        )
                        mri_summary = mri_content
                        mri_paths = self._get_mri_paths(filename)
                    category = self.handler.classify(patient_data, mri_content)
                elif mode == "tabular_mri":
                    # Text data + MRI images only (no parcellation text)
                    if self.config.mri_base_path:
                        mri_content = self.data_loader.get_mri_images_only(
                            filename, self.config.mri_base_path
                        )
                        mri_summary = self._summarize_mri_content(mri_content)
                        mri_paths = self._get_mri_paths(filename)
                    category = self.handler.classify(patient_data, mri_content)
                elif mode == "tabular_parcel_mri":
                    # Text data + MRI text + MRI images
                    if self.config.mri_base_path:
                        mri_content = self.data_loader.get_mri_content(
                            filename, self.config.mri_base_path, include_images=True
                        )
                        mri_summary = self._summarize_mri_content(mri_content)
                        mri_paths = self._get_mri_paths(filename)
                    category = self.handler.classify(patient_data, mri_content)
                elif mode == "parcel_mri":
                    # MRI data only (text + images)
                    if self.config.mri_base_path:
                        mri_content = self.data_loader.get_mri_content(
                            filename, self.config.mri_base_path, include_images=True
                        )
                        mri_summary = self._summarize_mri_content(mri_content)
                        mri_paths = self._get_mri_paths(filename)
                    category = self.handler.classify("", mri_content)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Save result with full paths
                record = {
                    "filename": filename,
                    "full_path": file_path,
                    "input": patient_data,
                    "output": category,
                    "timestamp": datetime.now().isoformat(),
                }
                if mri_summary:
                    record["mri_data_summary"] = mri_summary
                if mri_paths:
                    record["mri_paths"] = mri_paths

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{file_path}] -> {category}")
                if mri_paths:
                    print(f"  MRI: {mri_paths['subject_folder']}")

    def _summarize_mri_content(self, mri_content: List[Dict]) -> str:
        """Summarize MRI content for JSON serialization"""
        summary_parts = []
        for item in mri_content:
            if item["type"] == "text":
                summary_parts.append(item["text"])
            elif item["type"] == "image":
                summary_parts.append("[Image data included in processing]")
        return "\n".join(summary_parts)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Modular inference pipeline for patient classification using multimodal data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # InternVL Example
  python inference.py --txt_path /path/to/txt --mri_base_path /path/to/mri \\
    --model_name OpenGVLab/InternVL3_5-8B --mode tabular_parcel_mri

  # Ministral 3 Example
  python inference.py --txt_path /path/to/txt --mri_base_path /path/to/mri \\
    --model_name mistralai/Ministral-3-3B-Instruct-2512 --mode tabular_parcel_mri
""",
    )
    # Required arguments
    parser.add_argument(
        "--txt_path",
        type=str,
        required=True,
        help="Path to directory containing text files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., google/gemma-3-27b-it, OpenGVLab/InternVL3_5-8B)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "tabular",
            "tabular_parcel",
            "tabular_mri",
            "tabular_parcel_mri",
            "parcel_mri",
        ],
        help="""Inference mode:
        - tabular: Text data only
        - tabular_parcel: Text + MRI text data
        - tabular_mri: Text + MRI images only (no parcel text)
        - tabular_parcel_mri: Text + MRI text + MRI images
        - parcel_mri: MRI data only (text + images)""",
    )
    # Optional arguments
    parser.add_argument(
        "--mri_base_path",
        type=str,
        default=None,
        help="Path to MRI data directory (required for modes with MRI data)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL file path (default: auto-generated from model name and mode)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling for generation (default: False, uses greedy decoding)",
    )
    parser.add_argument(
        "--seed", type=int, default=666, help="Random seed (default: 666)"
    )
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments"""
    # Check if MRI path is required but not provided
    if args.mode in [
        "tabular_parcel",
        "tabular_mri",
        "tabular_parcel_mri",
        "parcel_mri",
    ]:
        if not args.mri_base_path:
            raise ValueError(
                f"Mode '{args.mode}' requires --mri_base_path to be specified"
            )

    # Check if paths exist
    if not os.path.exists(args.txt_path):
        raise FileNotFoundError(f"Text path does not exist: {args.txt_path}")
    if args.mri_base_path and not os.path.exists(args.mri_base_path):
        raise FileNotFoundError(f"MRI base path does not exist: {args.mri_base_path}")

    # Generate output file name if not provided
    if not args.output_file:
        model_basename = args.model_name.split("/")[-1].replace("-", "_")
        mri_basename = (
            args.mri_base_path.split("/")[-2] + "_" + args.mri_base_path.split("/")[-1]
            if args.mri_base_path
            else "no_mri"
        )
        txt_basename = (
            args.txt_path.split("/")[-2] + "_" + args.txt_path.split("/")[-1]
            if args.txt_path
            else "no_txt"
        )
        args.output_file = f"results_{model_basename}_mri_{mri_basename}_txt_{txt_basename}_{args.mode}.jsonl"
        print(f"Output file not specified. Using: {args.output_file}")

    return args


if __name__ == "__main__":
    # Parse and validate arguments
    args = parse_args()
    args = validate_args(args)

    # Set seed
    set_seed(args.seed)

    # Create configuration
    config = InferenceConfig(
        txt_path=args.txt_path,
        mri_base_path=args.mri_base_path,
        output_file=args.output_file,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
    )

    # Print configuration
    print("=" * 60)
    print("INFERENCE CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Text path: {config.txt_path}")
    print(f"MRI base path: {config.mri_base_path}")
    print(f"Output file: {config.output_file}")
    print(f"Max new tokens: {config.max_new_tokens}")
    print(f"Do sample: {config.do_sample}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    print()

    # Run pipeline
    print("Initializing model...")
    pipeline = InferencePipeline(config)
    print(f"Running inference in '{args.mode}' mode...")
    pipeline.run(mode=args.mode)

    print()
    print("=" * 60)
    print(f"Done! Results saved to: {config.output_file}")
    print("=" * 60)
