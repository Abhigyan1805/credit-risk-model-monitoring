"""The drafting model, behind a one-method interface so another model can be swapped in.

``Drafter.generate(messages) -> str`` takes chat messages (``[{"role", "content"}]``) and returns the
draft text. ``HFDrafter`` runs an open-weight instruct model locally with Hugging Face transformers
(no API, no key); ``ScriptedDrafter`` replays fixed drafts for tests. The default model and decoding
settings are recorded in every commentary JSON and in ``docs/AI_COMMENTARY.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class Drafter(Protocol):
    name: str

    def generate(self, messages: list[dict], attempt: int = 0) -> str: ...

    def settings(self) -> dict: ...


@dataclass
class HFSettings:
    model_id: str = DEFAULT_MODEL_ID
    quantization: str = "nf4"  # nf4 (bitsandbytes 4-bit, GPU) | fp16 (GPU) | fp32 (CPU)
    max_new_tokens: int = 400
    do_sample: bool = False  # first draft: greedy decoding
    retry_temperature: float = 0.7  # retries: seeded sampling, so a retry is not a copy of the rejected draft
    retry_top_p: float = 0.9
    repetition_penalty: float = 1.05
    seed: int = 0


class HFDrafter:
    """A local transformers chat model. Imports torch lazily so the rest of the package stays light."""

    def __init__(self, settings: HFSettings | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = settings or HFSettings()
        self.name = self.cfg.model_id
        self.revision = None
        local = Path(self.cfg.model_id)
        if (local / "HF_ID").exists():  # a pre-downloaded snapshot: record where it came from
            self.name = (local / "HF_ID").read_text().strip()
            rev = local / "REVISION"
            self.revision = rev.read_text().strip() if rev.exists() else None
        torch.manual_seed(self.cfg.seed)
        kwargs: dict = {}
        if self.cfg.quantization == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "cuda:0"
        elif self.cfg.quantization == "fp16":
            kwargs.update(dtype=torch.float16, device_map="cuda:0")
        else:
            kwargs.update(dtype=torch.float32, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, low_cpu_mem_usage=True, **kwargs)
        self.model.eval()
        self._torch = torch

    def generate(self, messages: list[dict], attempt: int = 0) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        sample = self.cfg.do_sample or attempt > 0
        decoding = {"do_sample": sample}
        if sample:
            self._torch.manual_seed(self.cfg.seed + attempt)  # reproducible retries
            decoding.update(temperature=self.cfg.retry_temperature, top_p=self.cfg.retry_top_p)
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                **decoding,
            )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def settings(self) -> dict:
        import transformers

        return {
            **asdict(self.cfg),
            "model_id": self.name,
            "loaded_from": self.cfg.model_id,
            "revision": self.revision,
            "device": torch_device_name(self._torch),
            "transformers": transformers.__version__,
            "torch": self._torch.__version__,
        }


def torch_device_name(torch) -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


class ScriptedDrafter:
    """Returns pre-written drafts in order (for tests and the corrupted-draft evaluation)."""

    def __init__(self, drafts: list[str], name: str = "scripted"):
        self.drafts = list(drafts)
        self.name = name
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict], attempt: int = 0) -> str:
        self.calls.append(messages)
        return self.drafts[min(len(self.calls) - 1, len(self.drafts) - 1)]

    def settings(self) -> dict:
        return {"model_id": self.name}
