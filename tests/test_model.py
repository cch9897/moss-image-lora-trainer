"""Vendored model tests: forward, checkpointing, upstream parity and diff allowlist."""

from __future__ import annotations

import difflib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from moss_lora.model import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
from tests.helpers import TINY_KWARGS, tiny_transformer

UPSTREAM_ENV = "MOSS_UPSTREAM_PATH"
UPSTREAM_DEFAULT = "/tmp/moss_upstream/pipeline_moss_image.py"


def upstream_path() -> Path:
    return Path(os.environ.get(UPSTREAM_ENV, UPSTREAM_DEFAULT))


def load_upstream_module():
    path = upstream_path()
    if not path.is_file():
        pytest.skip(f"upstream pipeline_moss_image.py not available at {path}")
    spec = importlib.util.spec_from_file_location("official_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_forward_shapes():
    model = tiny_transformer().eval()
    latents = torch.randn(2, 8, 16, 16)
    context = torch.randn(2, 5, 32)
    mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
    with torch.no_grad():
        out = model(latents, [250.0, 750.0], context, mask, return_dict=False)[0]
    assert out.shape == (2, 8, 16, 16)
    assert out.dtype == torch.float32


def test_padded_text_tokens_cannot_leak():
    model = tiny_transformer().eval()
    latents = torch.randn(1, 8, 8, 8)
    context = torch.randn(1, 6, 32)
    full = torch.ones(1, 6, dtype=torch.bool)
    masked = torch.tensor([[True] * 3 + [False] * 3])
    with torch.no_grad():
        base = model(latents, [300.0], context, masked, return_dict=False)[0]
        tweaked = context.clone()
        tweaked[:, 3:] += 10.0  # only padded positions change
        other = model(latents, [300.0], tweaked, masked, return_dict=False)[0]
    assert torch.equal(base, other)
    with torch.no_grad():
        full_out = model(latents, [300.0], context, full, return_dict=False)[0]
    assert not torch.equal(base, full_out)


def test_gradient_checkpointing_grad_parity():
    model = tiny_transformer().float()
    latents = torch.randn(2, 8, 8, 8)
    context = torch.randn(2, 4, 32)
    mask = torch.ones(2, 4, dtype=torch.bool)
    timesteps = [120.0, 880.0]

    def gradients(use_checkpointing: bool) -> list[torch.Tensor]:
        model.gradient_checkpointing = use_checkpointing
        model.train()
        for parameter in model.parameters():
            parameter.grad = None
        out = model(latents, timesteps, context, mask, return_dict=False)[0]
        out.square().mean().backward()
        return [parameter.grad.clone() for parameter in model.parameters() if parameter.grad is not None]

    plain = gradients(False)
    checkpointed = gradients(True)
    assert len(plain) == len(checkpointed) > 0
    for a, b in zip(plain, checkpointed):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_gradient_checkpointing_toggle():
    model = tiny_transformer()
    assert model.gradient_checkpointing is False
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing is True
    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False
    model.train()
    out = model(torch.randn(1, 8, 8, 8), [100.0], torch.randn(1, 3, 32), return_dict=False)[0]
    assert out.shape == (1, 8, 8, 8)


def test_vendored_matches_upstream_with_documented_changes():
    path = upstream_path()
    if not path.is_file():
        pytest.skip(f"upstream pipeline_moss_image.py not available at {path}")
    upstream = path.read_text(encoding="utf-8").splitlines()
    vendored_text = Path("moss_lora/model/moss_image.py").read_text(encoding="utf-8")
    vendored = vendored_text.splitlines()
    diff = list(difflib.unified_diff(upstream, vendored, lineterm="", n=0))
    added = {line[1:].strip() for line in diff if line.startswith("+") and not line.startswith("+++")}
    removed = {line[1:].strip() for line in diff if line.startswith("-") and not line.startswith("---")}

    assert removed == {
        "from transformers import AutoModel, AutoTokenizer",
        "x = layer(x, rope, mod[index:index + 1])",
        # Local mod #7: single fp32 cast in _RMSNorm.
        "y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)",
        # Local mod #8: memoized RoPE return.
        "return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)",
        # Local mod #9: timestep sinusoid factored into cached helpers.
        "times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)",
        "half = self.config.freq_dim // 2",
        "freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)",
        "phase = times.double()[:, None] * freq",
        "embedding = torch.cat((phase.cos(), phase.sin()), -1).float()",
        # Local mod #10: additive attn_mask=None pass-through (default stays identical).
        "def _attention(q, k, v, rope, heads):",
        "return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)",
        "def forward(self, x, rope):",
        "x = x + self.proj(_attention(q, k, v, rope, self.heads))",
        "x = block(x, rope)",
        "def forward(self, x, rope, modulation):",
        "x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads)) * gate_a[:, None]",
    }

    code_additions = {line for line in added if line and not line.startswith("#")}
    # Mod #10: every line inside the new batched-forward methods is allowed as a block.
    import ast

    mod10_lines: set[str] = set()
    for node in ast.walk(ast.parse(vendored_text)):
        if isinstance(node, ast.FunctionDef) and node.name in {
                "forward_batched", "enable_batched_forward", "disable_batched_forward",
                "_sdpa_varlen", "_sdpa_varlen_fwd", "_make_sac_context_fn", "_checkpoint_kwargs",
                "enable_selective_checkpointing", "disable_selective_checkpointing"}:
            mod10_lines.update(line.strip() for line in vendored[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.ClassDef) and node.name == "_VarlenSDPA":
            mod10_lines.update(line.strip() for line in vendored[node.lineno - 1:node.end_lineno])
    mod10_passthrough = {
        "def _attention(q, k, v, rope, heads, attn_mask=None):",
        "return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,",
        "attn_mask=attn_mask).transpose(1, 2).flatten(2)",
        "def forward(self, x, rope, attn_mask=None):",
        "x = x + self.proj(_attention(q, k, v, rope, self.heads, attn_mask))",
        "x = block(x, rope, attn_mask)",
        "def forward(self, x, rope, modulation, attn_mask=None):",
        "x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads,",
        "attn_mask)) * gate_a[:, None]",
        "self._batched_forward = False",
        "self._sac_policy = None",
        "layer, x, rope, mod[index:index + 1], **self._checkpoint_kwargs(False))",
        "if self._batched_forward:",
        "return self.forward_batched(hidden_states, timestep, encoder_hidden_states,",
        "encoder_attention_mask, return_dict, skip_layers)",
        "_sdpa_varlen = torch.compiler.allow_in_graph(_sdpa_varlen)",
        "from torch.nn.attention import SDPBackend, sdpa_kernel",
        "if (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)",
        "and attn_mask is not None and attn_mask.dtype == torch.bool",
        "and attn_mask.dim() == 4 and attn_mask.shape[1] == 1 and attn_mask.shape[2] == 1):",
        "return _sdpa_varlen(q.to(v.dtype), k.to(v.dtype), v, attn_mask.sum(-1).view(q.shape[0]))",
    }
    allowed_code = {
        "import torch.utils.checkpoint",
        "self.gradient_checkpointing = False",
        "def enable_gradient_checkpointing(self):",
        "self.gradient_checkpointing = True",
        "def disable_gradient_checkpointing(self):",
        "from transformers import AutoModel, AutoTokenizer",
        "if self.training and self.gradient_checkpointing:",
        "x = torch.utils.checkpoint.checkpoint(",
        "layer, x, rope, mod[index:index + 1], use_reentrant=False)",
        "else:",
        "x = layer(x, rope, mod[index:index + 1])",
        # Local mod #7: single fp32 cast in _RMSNorm.
        "xf = x.float()",
        "y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)",
        # Local mod #8: RoPE LRU cache.
        "from collections import OrderedDict",
        "self._rope_cache = OrderedDict()",
        "if hasattr(self, '_rope_cache'):",
        "self._rope_cache.clear()",
        "key = (height, width, text_length, str(device))",
        "cached = self._rope_cache.get(key)",
        "if cached is not None:",
        "self._rope_cache.move_to_end(key)",
        "return cached",
        "rope = torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)",
        "self._rope_cache[key] = rope",
        "while len(self._rope_cache) > 16:",
        "self._rope_cache.popitem(last=False)",
        "return rope",
        # Local mod #9: cached fp64 frequencies + integer-timestep table.
        "self._freq_cache = {}",
        "self._time_table_cache = {}",
        "def _time_freq(self, device):",
        "key = str(device)",
        "freq = self._freq_cache.get(key)",
        "if freq is None:",
        "half = self.config.freq_dim // 2",
        "freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)",
        "self._freq_cache[key] = freq",
        "return freq",
        "def _time_embedding_input(self, times, device):",
        "if times.device.type == 'cpu':",
        "rounded = times.round()",
        "if bool(((times == rounded) & (rounded >= 0) & (rounded <= 1000)).all()):",
        "table = self._time_table_cache.get(key)",
        "if table is None:",
        "steps = torch.arange(1001, dtype=torch.float32).double()[:, None].to(device)",
        "phase = steps * self._time_freq(device)",
        "table = torch.cat((phase.cos(), phase.sin()), -1).float()",
        "self._time_table_cache[key] = table",
        "return table[rounded.long().to(device)]",
        "times = times.to(device)",
        "freq = self._time_freq(times.device)",
        "phase = times.double()[:, None] * freq",
        "return torch.cat((phase.cos(), phase.sin()), -1).float()",
        "times = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1)",
        "embedding = self._time_embedding_input(times, hidden_states.device)",
    }
    assert code_additions <= allowed_code | mod10_lines | mod10_passthrough
    assert len(code_additions) >= 10


def test_official_vs_vendored_parity():
    official = load_upstream_module()
    torch.manual_seed(11)
    latents = torch.randn(1, 8, 8, 8)
    context = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    timesteps = [333.0]

    torch.manual_seed(123)
    theirs = official.MOSSImageTransformer(**TINY_KWARGS).eval()
    torch.manual_seed(123)
    ours = MOSSImageTransformer(**TINY_KWARGS).eval()
    for (name_a, param_a), (name_b, param_b) in zip(theirs.named_parameters(), ours.named_parameters()):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)

    with torch.no_grad():
        out_theirs = theirs(latents, timesteps, context, mask, return_dict=False)[0]
        out_ours = ours(latents, timesteps, context, mask, return_dict=False)[0]
    assert torch.equal(out_theirs, out_ours)

    sched_theirs = official.MOSSImageScheduler()
    sched_ours = MOSSImageScheduler()
    sched_theirs.set_timesteps(7)
    sched_ours.set_timesteps(7)
    assert torch.equal(sched_theirs.timesteps, sched_ours.timesteps)
    assert torch.equal(sched_theirs.sigmas, sched_ours.sigmas)

    torch.manual_seed(7)
    embeds = torch.randn(1, 4, 32)
    torch.manual_seed(7)
    neg = torch.randn(1, 4, 32)
    pipe_theirs = official.MOSSImagePipeline(transformer=theirs, scheduler=sched_theirs,
                                             text_encoder=None, tokenizer=None, vae=None)
    pipe_ours = MOSSImagePipeline(transformer=ours, scheduler=sched_ours,
                                  text_encoder=None, tokenizer=None, vae=None)
    pipe_theirs.set_progress_bar_config(disable=True)
    pipe_ours.set_progress_bar_config(disable=True)
    # Explicit latents + explicit negative embeds keep both runs RNG-free and identical.
    kwargs = dict(prompt_embeds=embeds, negative_prompt_embeds=neg, height=128, width=128,
                  num_inference_steps=3, guidance_scale=3.0, uncond_skip_layers=[],
                  output_type="latent", latents=latents.clone())
    lat_theirs = pipe_theirs(**kwargs).images
    lat_ours = pipe_ours(**kwargs).images
    assert torch.equal(lat_theirs, lat_ours)


def test_scheduler_defaults_match_official():
    official = load_upstream_module()
    sched = MOSSImageScheduler()
    assert sched.config.shift == 5.0
    assert sched.config.training_shift == 1.0
    assert sched.config.num_train_timesteps == 1000
    theirs = official.MOSSImageScheduler()
    assert theirs.config.shift == sched.config.shift
    assert theirs.config.training_shift == sched.config.training_shift


def test_module_imports_without_transformers():
    """Importing the vendored module and running it must not require transformers."""
    code = (
        "import importlib.util\n"
        "import sys\n"
        "real_find_spec = importlib.util.find_spec\n"
        "def find_spec(name, package=None):\n"
        "    if name == 'transformers' or name.startswith('transformers.'):\n"
        "        return None  # diffusers availability probes see it absent\n"
        "    return real_find_spec(name, package)\n"
        "importlib.util.find_spec = find_spec\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'transformers' or name.startswith('transformers.'):\n"
        "            raise ImportError('transformers is blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "try:\n"  # positive control: prove the blocker is real
        "    import transformers  # noqa: F401\n"
        "    raise SystemExit('BLOCKER FAILED: transformers imported')\n"
        "except ImportError:\n"
        "    pass\n"
        "import torch\n"
        "from moss_lora.model import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer\n"
        "m = MOSSImageTransformer(dim=32, ffn_dim=64, num_heads=4, num_layers=1,\n"
        "                         text_dim=16, freq_dim=16, latent_channels=8).eval()\n"
        "out = m(torch.randn(1, 8, 8, 8), [100.0], torch.randn(1, 5, 16), return_dict=False)[0]\n"
        "assert out.shape == (1, 8, 8, 8)\n"
        "pipe = MOSSImagePipeline(transformer=m, scheduler=MOSSImageScheduler(),\n"
        "                         text_encoder=None, tokenizer=None, vae=None)\n"
        "pipe.set_progress_bar_config(disable=True)\n"
        "img = pipe(prompt_embeds=torch.randn(1, 5, 16), negative_prompt_embeds=torch.randn(1, 5, 16),\n"
        "           height=128, width=128, num_inference_steps=2, guidance_scale=3.0,\n"
        "           uncond_skip_layers=[], output_type='latent').images\n"
        "assert img.shape == (1, 8, 8, 8)\n"
        "print('NO_TRANSFORMERS_OK')\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True,
                            capture_output=True, text=True, timeout=300)
    assert "NO_TRANSFORMERS_OK" in result.stdout
