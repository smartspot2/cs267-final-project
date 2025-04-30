from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from outlines.models.transformers_vision import transformers_vision
from pydantic import BaseModel
import outlines
import gc
import torch
from PIL import Image
from typing import Union
import utils.device
import utils.cache

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from .base import Verifier

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Optional device map that one can use to let `transformers` share a single GPU and CPU.
DEVICE_MAP = {
    "visual": 1,
    "model.embed_tokens": 1,
    "model.layers.0": 1,
    "model.layers.1": 1,
    "model.layers.2": 1,
    "model.layers.3": 1,
    "model.layers.4": 1,
    "model.layers.5": 1,
    "model.layers.6": 1,
    "model.layers.7": 1,
    "model.layers.8": 1,
    "model.layers.9": 1,
    "model.layers.10": 1,
    "model.layers.11": "cpu",
    "model.layers.12": "cpu",
    "model.layers.13": "cpu",
    "model.layers.14": "cpu",
    "model.layers.15": "cpu",
    "model.layers.16": "cpu",
    "model.layers.17": "cpu",
    "model.layers.18": "cpu",
    "model.layers.19": "cpu",
    "model.layers.20": "cpu",
    "model.layers.21": "cpu",
    "model.layers.22": "cpu",
    "model.layers.23": "cpu",
    "model.layers.24": "cpu",
    "model.layers.25": "cpu",
    "model.layers.26": "cpu",
    "model.layers.27": "cpu",
    "model.norm": "cpu",
    "model.rotary_emb": "cpu",
    "lm_head": "cpu",
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VERIFIER_PROMPT_PATH = os.path.join(SCRIPT_DIR, "verifier_prompt.txt")

@staticmethod
def load_verifier_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as f:
        return f.read()

class Score(BaseModel):
    explanation: str
    score: float


class Grading(BaseModel):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    overall_score: Score


class QwenVerifier(Verifier):
    SUPPORTED_METRIC_CHOICES = [
        "accuracy_to_prompt",
        "creativity_and_originality",
        "visual_quality_and_realism",
        "consistency_and_cohesion",
        "emotional_or_thematic_resonance",
        "overall_score",
    ]

    def __init__(self, seed=1994, model_name=DEFAULT_QWEN_MODEL_ID, **kwargs):
        # super().__init__(seed=seed, prompt_path=kwargs.pop("prompt_path", None))
        self.seed = seed
        model, processor = self.load_verifier(model_name)

        model_kwargs = self._prepare_model_kwargs(**kwargs)
        prompt_path = kwargs.pop("prompt_path", None)
        prompt_path = prompt_path or DEFAULT_VERIFIER_PROMPT_PATH
        self.verifier_prompt = load_verifier_prompt(prompt_path)


        print(utils.device.DEVICE)
        self.model = transformers_vision(
            DEFAULT_QWEN_MODEL_ID,
            model_class=model.__class__,
            device=utils.device.DEVICE,
            model_kwargs=model_kwargs,
            processor_class=processor.__class__,
        )
        self.structured_generator = outlines.generate.json(self.model, Grading)

        del model, processor
        gc.collect()

        self.max_new_tokens = kwargs.get("max_new_tokens", 800)

    @torch.no_grad()
    def load_verifier(self, model_name: str):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, cache_dir=utils.cache.CACHE_DIR)
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=utils.cache.CACHE_DIR)
        return model, processor

    def prepare_conversations(self, prompt):
        user_content = []
        conversation = [
            {"role": "system", "content": self.verifier_prompt},
        ]
        user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})
        user_content = {"role": "user", "content": user_content}
        conversation.append(user_content)
        return conversation

    def prepare_inputs(self, images: Union[list[Image.Image], Image.Image], prompts: Union[list[str], str]) -> dict:
        assert len(images) == len(prompts)

        conversations = []
        for prompt in prompts:
            conversations.append(self.prepare_conversations(prompt))

        assert len(conversations) == len(images) == len(prompts)

        prompts = [self.model.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
        images = [[image] for image in images]
        inputs = {"images": images, "prompts": prompts}
        return inputs

    @torch.no_grad()
    def get_reward(self, prompts, images) -> list[dict[str, float]]:
        inputs = self.prepare_inputs(images, prompts)
        # TODO: might need to iterate `inputs` in batches depending on the resources.
        outputs = self.structured_generator(
            inputs["prompts"], inputs["images"], max_tokens=self.max_new_tokens, seed=self.seed
        )
        outputs = [o.dict() for o in outputs]
        return outputs

    def _prepare_model_kwargs(self, **kwargs):
        model_kwargs = {"torch_dtype": torch.bfloat16}
        use_low_gpu_vram = kwargs.get("use_low_gpu_vram", False)
        if not use_low_gpu_vram:
            model_kwargs.update({"attn_implementation": "flash_attention_2"})
        else:
            model_kwargs.update({"device_map": "auto"})
        return model_kwargs