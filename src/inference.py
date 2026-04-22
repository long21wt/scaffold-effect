import argparse
import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T  # noqa: N812
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
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

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    txt_path: str
    mri_base_path: str | None = None
    output_file: str = "results.jsonl"
    model_name: str = "google/gemma-3-27b-it"
    max_new_tokens: int = 4096
    do_sample: bool = False
    categories: list[str] = None

    def __post_init__(self) -> None:
        if self.categories is None:
            self.categories = [
                "Major Depressive Disorder",
                "Control (no disorder detected)",
            ]


class DataLoader:
    @staticmethod
    def load_text_file(file_path: str | Path) -> str:
        return Path(file_path).read_text(encoding="utf-8")

    @staticmethod
    def get_mri_content(
        txt_filename: str, mri_base_path: str, include_images: bool = False
    ) -> list[dict[str, Any]]:
        sub_folder = f"sub-{int(Path(txt_filename).stem):04d}"
        subject_path = Path(mri_base_path) / sub_folder

        if not subject_path.exists():
            return [{"type": "text", "text": f"No MRI data found for {sub_folder}"}]

        content_items = []
        for session_path in sorted(subject_path.iterdir()):
            if not session_path.is_dir():
                continue
            content_items.append(
                {"type": "text", "text": f"\n=== {session_path.name} ==="}
            )
            for fp in sorted(session_path.iterdir()):
                if fp.suffix == ".txt":
                    content = DataLoader.load_text_file(fp)
                    content_items.append(
                        {"type": "text", "text": f"\n{fp.name}:\n{content}"}
                    )
                elif fp.suffix == ".png" and include_images:
                    content_items.append({"type": "image", "image": Image.open(fp)})
                    content_items.append(
                        {"type": "text", "text": f"[Image: {fp.name}]"}
                    )

        return (
            content_items
            if content_items
            else [{"type": "text", "text": "No MRI data found"}]
        )

    @staticmethod
    def get_mri_images_only(
        txt_filename: str, mri_base_path: str
    ) -> list[dict[str, Any]]:
        sub_folder = f"sub-{int(Path(txt_filename).stem):04d}"
        subject_path = Path(mri_base_path) / sub_folder

        if not subject_path.exists():
            return [{"type": "text", "text": f"No MRI data found for {sub_folder}"}]

        content_items = []
        for session_path in sorted(subject_path.iterdir()):
            if not session_path.is_dir():
                continue
            content_items.append(
                {"type": "text", "text": f"\n=== {session_path.name} ==="}
            )
            for fp in sorted(session_path.iterdir()):
                if fp.suffix == ".png":
                    content_items.append({"type": "image", "image": Image.open(fp)})
                    content_items.append(
                        {"type": "text", "text": f"[Image: {fp.name}]"}
                    )

        return (
            content_items
            if content_items
            else [{"type": "text", "text": "No MRI images found"}]
        )

    @staticmethod
    def get_mri_text_only(txt_filename: str, mri_base_path: str) -> str:
        content_items = DataLoader.get_mri_content(
            txt_filename, mri_base_path, include_images=False
        )
        text_parts = [item["text"] for item in content_items if item["type"] == "text"]
        return "\n".join(text_parts) if text_parts else "No MRI text data found"


class ModelHandler(ABC):
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        pass

    @abstractmethod
    def classify(self, text: str, mri_content: Any) -> str:  # noqa: ANN401
        pass

    def _build_prompt(
        self,
        text: str,
        mri_data: str | None = None,
        include_patient_data: bool = True,
    ) -> str:
        prompt_parts = ["You are given patient"]
        if include_patient_data:
            prompt_parts.append("clinical information and The weather is sunny today")
        if mri_data:
            prompt_parts.append(
                "and their MRI data "
                "(brain parcellation volume, visualization of brain regions)"
            )

        prompt = " ".join(prompt_parts) + ".\n"
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

    def _load_model(self) -> None:
        logger.info("loading %s", self.config.model_name)
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

    @staticmethod
    def build_transform(input_size: int) -> T.Compose:
        mean, std = InternVLHandler.IMAGENET_MEAN, InternVLHandler.IMAGENET_STD
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: list,
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
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
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ) -> list[Image.Image]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
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

    def _process_image(
        self, image_obj: Image.Image, input_size: int = 448, max_num: int = 12
    ) -> torch.Tensor:
        image = image_obj.convert("RGB")
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(img) for img in images]
        return torch.stack(pixel_values)

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: ANN401
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

        generation_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }

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

    def _load_model(self) -> None:
        # Lazy import to avoid crashing older environments used for InternVL
        try:
            from transformers import (  # noqa: PLC0415
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )
        except ImportError as e:
            raise ImportError(  # noqa: TRY003
                "MistralCommonBackend not found. "
                "You are likely running in the 'InternVL' environment. "
                "To use Ministral, switch to a newer environment with transformers."
            ) from e

        logger.info("loading %s", self.config.model_name)
        self.processor = MistralCommonBackend.from_pretrained(self.config.model_name)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.config.model_name, device_map="auto"
        )

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: C901, ANN401
        prompt_text = self._build_prompt(
            text,
            "multimodal" if isinstance(mri_content, list) else mri_content,
            include_patient_data=bool(text),
        )

        user_content_list = [{"type": "text", "text": prompt_text}]
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    user_content_list.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    # MistralCommonBackend requires image_url with data URI, not raw PIL
                    data_url = self._image_to_data_url(item["image"])
                    user_content_list.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )

        messages = [{"role": "user", "content": user_content_list}]
        tokenized = self.processor.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        )

        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.to(self.model.device)

        if "pixel_values" in tokenized:
            tokenized["pixel_values"] = tokenized["pixel_values"].to(
                dtype=torch.bfloat16, device=self.model.device
            )

        # image_sizes must match pixel_values spatial dims; model uses this to unpad
        # tiles
        image_sizes = None
        if "pixel_values" in tokenized:
            h, w = tokenized["pixel_values"].shape[-2:]
            image_sizes = [(h, w)] * tokenized["pixel_values"].shape[0]

        with torch.inference_mode():
            output = self.model.generate(
                **tokenized,
                image_sizes=image_sizes,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )[0]

        return self.processor.decode(output[len(tokenized["input_ids"][0]) :]).strip()


class PixtralHandler(ModelHandler):
    """Handler for Pixtral (Mistral Vision) models"""

    def _load_model(self) -> None:
        logger.info("loading %s", self.config.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: ANN401
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

    def _load_model(self) -> None:
        model_id = self.config.model_name
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: ANN401
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        if isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        prompt = self._build_prompt(text)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",  # noqa: E501
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
                        "text": "You are a helpful medical assistant in clinical psychiatry.",  # noqa: E501
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _classify_with_multimodal(
        self, text: str, mri_content_items: list[dict]
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
                        "text": "You are a helpful medical assistant in clinical psychiatry.",  # noqa: E501
                    }
                ],
            },
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages)

    def _generate(self, messages: list[dict]) -> str:
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


_SYSTEM_MSG = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "You are a helpful medical assistant in clinical psychiatry.",
        }
    ],
}


class _QwenVLUtilsBase(ModelHandler):
    """Shared logic for Qwen2-VL and Qwen2.5-VL (vision processed via qwen_vl_utils)."""

    process_vision_info: Any = None

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: ANN401
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        if isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        prompt = self._build_prompt(text)
        return self._generate(
            [
                _SYSTEM_MSG,
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        )

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        return self._generate(
            [
                _SYSTEM_MSG,
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        )

    def _classify_with_multimodal(
        self, text: str, mri_content_items: list[dict]
    ) -> str:
        prompt_text = self._build_prompt(
            text, "multimodal", include_patient_data=bool(text)
        )
        user_content = [{"type": "text", "text": prompt_text}, *list(mri_content_items)]
        return self._generate([_SYSTEM_MSG, {"role": "user", "content": user_content}])

    def _generate(self, messages: list[dict]) -> str:
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        image_inputs, video_inputs = self.process_vision_info(messages)  # pylint: disable=not-callable
        inputs = self.processor(  # pylint: disable=not-callable
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
        return self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        ).strip()


class Qwen2VLHandler(_QwenVLUtilsBase):
    """Handler for Qwen 2 VL models"""

    def _load_model(self) -> None:
        from qwen_vl_utils import process_vision_info  # noqa: PLC0415

        self.process_vision_info = process_vision_info
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)


class Qwen2_5VLHandler(_QwenVLUtilsBase):  # noqa: N801
    """Handler for Qwen 2.5 VL models"""

    def _load_model(self) -> None:
        from qwen_vl_utils import process_vision_info  # noqa: PLC0415

        self.process_vision_info = process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)


class Qwen2_5Handler(ModelHandler):  # noqa: N801
    def _load_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoTokenizer.from_pretrained(self.config.model_name)

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: ANN401
        if isinstance(mri_content, list):
            raise TypeError("Qwen2.5 model does not support multimodal input.")  # noqa: TRY003
        if isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        prompt = self._build_prompt(text)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful medical assistant in clinical psychiatry.",  # noqa: E501
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
                        "text": "You are a helpful medical assistant in clinical psychiatry.",  # noqa: E501
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._generate(messages)

    def _generate(self, messages: list[dict]) -> str:
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


class _QwenVLDirectBase(ModelHandler):
    """Shared logic for Qwen3-VL and Qwen3-VL-MoE (apply_chat_template with tokenize=True)."""  # noqa: E501

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: ANN401
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        if isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
        return self._classify_text_only(text)

    def _classify_text_only(self, text: str) -> str:
        prompt = self._build_prompt(text)
        return self._generate(
            [
                _SYSTEM_MSG,
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        )

    def _classify_with_text_only(self, text: str, mri_data: str) -> str:
        prompt = self._build_prompt(text, mri_data)
        return self._generate(
            [
                _SYSTEM_MSG,
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        )

    def _classify_with_multimodal(
        self, text: str, mri_content_items: list[dict]
    ) -> str:
        prompt_text = self._build_prompt(
            text, "multimodal", include_patient_data=bool(text)
        )
        user_content = [{"type": "text", "text": prompt_text}, *list(mri_content_items)]
        return self._generate([_SYSTEM_MSG, {"role": "user", "content": user_content}])

    def _generate(self, messages: list[dict]) -> str:
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
        return self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        ).strip()


class Qwen3VLHandler(_QwenVLDirectBase):
    """Handler for Qwen 3 VL models"""

    def _load_model(self) -> None:
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)


class Qwen3VLMoeHandler(_QwenVLDirectBase):
    """Handler for Qwen 3 VL MoE models"""

    def _load_model(self) -> None:
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)


class LlavaOneVisionHandler(ModelHandler):
    """Handler for LLaVA-One-Vision models"""

    def _load_model(self) -> None:
        logger.info("loading %s", self.config.model_name)
        # Import process_vision_info locally as required by the snippet
        from qwen_vl_utils import process_vision_info  # noqa: PLC0415

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

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: ANN401
        if isinstance(mri_content, list):
            return self._classify_with_multimodal(text, mri_content)
        if isinstance(mri_content, str):
            return self._classify_with_text_only(text, mri_content)
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
        self, text: str, mri_content_items: list[dict]
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

    def _generate(self, messages: list[dict]) -> str:
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
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()


class Glm4vHandler(ModelHandler):
    """Handler for GLM-4V models"""

    def _load_model(self) -> None:
        logger.info("loading %s", self.config.model_name)
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, use_fast=True
        )

    def classify(self, text: str, mri_content: Any = None) -> str:  # noqa: ANN401
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
                if k in {"images", "pixel_values"}:
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
    @staticmethod
    def create_handler(config: InferenceConfig) -> ModelHandler:  # noqa: PLR0911, C901
        model_name = config.model_name.lower()

        if "internvl" in model_name:
            return InternVLHandler(config)
        if "ministral" in model_name or (
            "mistral" in model_name and "2512" in model_name
        ):
            return MinistralHandler(config)
        if "glm" in model_name:
            return Glm4vHandler(config)
        if "llava-onevision" in model_name:
            return LlavaOneVisionHandler(config)
        if "pixtral" in model_name:
            return PixtralHandler(config)
        if "gemma" in model_name or "medgemma" in model_name:
            return GemmaHandler(config)
        if "qwen2-vl" in model_name:
            return Qwen2VLHandler(config)
        if "qwen2.5-vl" in model_name or "qwen2_5_vl" in model_name:
            return Qwen2_5VLHandler(config)
        if "qwen3-vl" in model_name and (
            "moe" in model_name or "a3b" in model_name or "a22b" in model_name
        ):
            return Qwen3VLMoeHandler(config)
        if "qwen3-vl" in model_name:
            return Qwen3VLHandler(config)
        if "qwen2.5" in model_name or "qwen2_5" in model_name:
            return Qwen2_5Handler(config)
        raise ValueError(f"Unsupported model: {config.model_name}")  # noqa: TRY003


class InferencePipeline:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.handler = ModelFactory.create_handler(config)
        self.data_loader = DataLoader()

    def _get_mri_paths(self, txt_filename: str) -> dict[str, Any]:
        if not self.config.mri_base_path:
            return None

        sub_folder = f"sub-{int(Path(txt_filename).stem):04d}"
        subject_path = Path(self.config.mri_base_path) / sub_folder

        if not subject_path.exists():
            return None

        mri_paths = {"subject_folder": str(subject_path), "sessions": {}}

        for session_path in sorted(subject_path.iterdir()):
            if not session_path.is_dir():
                continue
            session_folder = session_path.name
            mri_paths["sessions"][session_folder] = {
                "session_path": str(session_path),
                "files": [],
            }
            for fp in sorted(session_path.iterdir()):
                mri_paths["sessions"][session_folder]["files"].append(str(fp))

        return mri_paths

    def run(self, mode: str = "tabular") -> None:  # noqa: PLR0912, C901
        with Path(self.config.output_file).open("w", encoding="utf-8") as out_f:
            for fp in tqdm(sorted(Path(self.config.txt_path).iterdir())):
                if fp.suffix.lower() != ".txt":
                    continue
                filename = fp.name
                patient_data = self.data_loader.load_text_file(fp)

                # Determine what MRI data to include
                mri_content = None
                mri_summary = None
                mri_paths = None

                if mode == "tabular":
                    category = self.handler.classify(patient_data, None)
                elif mode == "tabular_parcel":
                    if self.config.mri_base_path:
                        mri_content = self.data_loader.get_mri_text_only(
                            filename, self.config.mri_base_path
                        )
                        mri_summary = mri_content
                        mri_paths = self._get_mri_paths(filename)
                    category = self.handler.classify(patient_data, mri_content)
                elif mode == "tabular_mri":
                    if self.config.mri_base_path:
                        mri_content = self.data_loader.get_mri_images_only(
                            filename, self.config.mri_base_path
                        )
                        mri_summary = self._summarize_mri_content(mri_content)
                        mri_paths = self._get_mri_paths(filename)
                    category = self.handler.classify(patient_data, mri_content)
                elif mode == "tabular_parcel_mri":
                    if self.config.mri_base_path:
                        mri_content = self.data_loader.get_mri_content(
                            filename, self.config.mri_base_path, include_images=True
                        )
                        mri_summary = self._summarize_mri_content(mri_content)
                        mri_paths = self._get_mri_paths(filename)
                    category = self.handler.classify(patient_data, mri_content)
                elif mode == "parcel_mri":
                    if self.config.mri_base_path:
                        mri_content = self.data_loader.get_mri_content(
                            filename, self.config.mri_base_path, include_images=True
                        )
                        mri_summary = self._summarize_mri_content(mri_content)
                        mri_paths = self._get_mri_paths(filename)
                    category = self.handler.classify("", mri_content)
                else:
                    raise ValueError(f"Unknown mode: {mode}")  # noqa: TRY003

                record = {
                    "filename": filename,
                    "full_path": str(fp),
                    "input": patient_data,
                    "output": category,
                    "timestamp": datetime.now().isoformat(),
                }
                if mri_summary:
                    record["mri_data_summary"] = mri_summary
                if mri_paths:
                    record["mri_paths"] = mri_paths

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info("%s -> %s", fp.name, category)
                if mri_paths:
                    logger.info("  MRI: %s", mri_paths['subject_folder'])

    def _summarize_mri_content(self, mri_content: list[dict]) -> str:
        summary_parts = []
        for item in mri_content:
            if item["type"] == "text":
                summary_parts.append(item["text"])
            elif item["type"] == "image":
                summary_parts.append("[Image data included in processing]")
        return "\n".join(summary_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular inference pipeline for patient classification using multimodal data",  # noqa: E501
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
        help="Model name or path (e.g., google/gemma-3-27b-it, OpenGVLab/InternVL3_5-8B)",  # noqa: E501
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
        help="Output JSONL file path (default: auto-generated from model name and mode)",  # noqa: E501
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


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode in [
        "tabular_parcel",
        "tabular_mri",
        "tabular_parcel_mri",
        "parcel_mri",
    ] and not args.mri_base_path:
        raise ValueError(  # noqa: TRY003
            f"Mode '{args.mode}' requires --mri_base_path to be specified"
        )

    if not Path(args.txt_path).exists():
        raise FileNotFoundError(f"Text path does not exist: {args.txt_path}")  # noqa: TRY003
    if args.mri_base_path and not Path(args.mri_base_path).exists():
        raise FileNotFoundError(f"MRI base path does not exist: {args.mri_base_path}")  # noqa: TRY003

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
        args.output_file = (
            f"results_{model_basename}_mri_{mri_basename}"
            f"_txt_{txt_basename}_{args.mode}.jsonl"
        )
        logger.info("output: %s", args.output_file)

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args = validate_args(args)

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

    logger.info(
        "model=%s mode=%s out=%s", config.model_name, args.mode, config.output_file
    )

    pipeline = InferencePipeline(config)
    pipeline.run(mode=args.mode)

    logger.info("saved %s", config.output_file)
