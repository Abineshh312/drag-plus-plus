"""src.models.student

DRAG++ Student Model wrapper.

Design goals:
- Run on Google Colab Free (T4, ~15GB VRAM)
- Prefer Qwen3.5-2B (your choice), fall back automatically if it fails
- Support 4-bit QLoRA-friendly loading (NF4) for training/inference efficiency

Note: Training code (LoRA/QLoRA) lives in scripts/train.py and notebooks.
This module focuses on robust model loading + generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass(frozen=True)
class StudentLoadSpec:
    """A single model load candidate."""

    key: str
    hf_id: str


class StudentModel:
    """Lightweight student model wrapper for retrieval-conditioned generation."""

    # Ordered by preference (we will try these in order when model_name="auto")
    DEFAULT_CANDIDATES: Sequence[StudentLoadSpec] = (
        # Preferred for Colab free tier: strong but still feasible with 4-bit QLoRA
        StudentLoadSpec("qwen3.5-1.5b", "Qwen/Qwen3.5-1.5B-Instruct"),
        StudentLoadSpec("qwen3.5-2b", "Qwen/Qwen3.5-2B-Instruct"),
        # Proven-available fallbacks
        StudentLoadSpec("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
        StudentLoadSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    )

    def __init__(
        self,
        model_name: str = "auto",
        device: Optional[str] = None,
        quantization: str = "4bit",  # "4bit" | "8bit" | "none"
        torch_dtype: Optional[torch.dtype] = None,
        max_context_tokens: int = 2048,
    ) -> None:
        """Initialize and load a student model.

        Args:
            model_name: "auto" (recommended) or a key like "qwen3.5-2b".
            device: "cuda" or "cpu". Auto-detected if None.
            quantization: Prefer "4bit" for Colab Free. Use "none" for CPU.
            torch_dtype: dtype override. Defaults: fp16 on cuda, fp32 on cpu.
            max_context_tokens: tokenizer/model max context (soft cap for prompts).
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = quantization
        self.max_context_tokens = max_context_tokens

        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        # Resolve candidate list
        if model_name == "auto":
            candidates = list(self.DEFAULT_CANDIDATES)
        else:
            # allow direct HF id as well
            spec = next((c for c in self.DEFAULT_CANDIDATES if c.key == model_name), None)
            candidates = [spec] if spec else [StudentLoadSpec(model_name, model_name)]

        last_err: Optional[Exception] = None
        for spec in candidates:
            try:
                self._load(spec)
                logger.info(f"Student model ready: {spec.key} ({spec.hf_id})")
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(f"Failed to load {spec.key} ({spec.hf_id}): {e}")

        raise RuntimeError(
            "Unable to load any student model candidates. "
            "Try setting quantization='none' on CPU or choose a smaller model."
        ) from last_err

    def _bnb_config(self) -> Optional[BitsAndBytesConfig]:
        if self.device != "cuda":
            return None

        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        if self.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        return None

    def _load(self, spec: StudentLoadSpec) -> None:
        logger.info(f"Loading student: {spec.hf_id} on {self.device} (quant={self.quantization})")

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)

        bnb = self._bnb_config()
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=self.torch_dtype,
            quantization_config=bnb,
        )

        self.model.eval()
        self.model_key = spec.key
        self.hf_id = spec.hf_id

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Loaded {spec.key}: {n_params/1e6:.1f}M params")

    def generate(
        self,
        query: str,
        context: str,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate an answer grounded in provided context."""
        prompt = self._format_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_context_tokens)

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _format_prompt(self, query: str, context: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant. Answer ONLY using the provided evidence. "
            "If evidence is insufficient, say you don't know.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\nEvidence:\n{context}\n\nQuestion: {query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
