"""Frozen Qwen3.5 text-encoder loading and the official per-prompt encoding.

transformers is imported lazily so that tiny CPU tests (and every code path that
only touches cached embeddings) need only torch + diffusers.

The encoding must match the official pipeline byte-for-byte:
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                              tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
    out = text_encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
"""

from __future__ import annotations

import gc

import torch


class TextEncoder:
    def __init__(self, model, tokenizer, device: torch.device | str, max_length: int = 4096):
        self.device = torch.device(device)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, prompt: str, max_length: int | None = None) -> torch.Tensor:
        """prompt -> [L, D] hidden states on self.device (model dtype); no padding."""
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True,
                                max_length=max_length or self.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs, use_cache=False, return_dict=True)
        return output.last_hidden_state[0]

    def unload(self) -> None:
        """Drop the model and free GPU memory (call before loading the VAE)."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_text_encoder(path: str, device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16,
                      local_files_only: bool = False, revision: str | None = None,
                      max_length: int = 4096, subfolder: str | None = None,
                      tokenizer_path: str | None = None) -> TextEncoder:
    """Load the frozen text encoder; tokenizer files may live in a separate directory."""
    from transformers import AutoModel, AutoTokenizer  # lazy import: keeps CPU test deps minimal

    tokenizer_kwargs: dict = {"local_files_only": local_files_only}
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or path, **tokenizer_kwargs)
    model_kwargs: dict = dict(tokenizer_kwargs, dtype=dtype)
    if subfolder:
        model_kwargs["subfolder"] = subfolder
    model = AutoModel.from_pretrained(path, **model_kwargs).eval()
    return TextEncoder(model, tokenizer, device, max_length)
