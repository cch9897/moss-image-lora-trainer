"""Latent-space math tests: patchify inverse, BatchNorm normalization, decode parity."""

from __future__ import annotations

import pytest
import torch

from moss_lora.model import MOSSImagePipeline, MOSSImageScheduler
from moss_lora.utils import (
    decode_image,
    encode_image_latent,
    patchify_latent,
    unpatchify_latent,
    vae_bn_stats,
)
from tests.helpers import FakeVae, tiny_transformer


def test_patchify_unpatchify_inverse_real_dims():
    torch.manual_seed(0)
    raw = torch.randn(3, 32, 12, 16)  # [B, VAE channels, 2h, 2w]
    latent = patchify_latent(raw)
    assert latent.shape == (3, 128, 6, 8)
    assert torch.equal(unpatchify_latent(latent), raw)

    back = patchify_latent(unpatchify_latent(latent))
    assert torch.equal(back, latent)


def test_patchify_unpatchify_inverse_tiny_dims():
    torch.manual_seed(1)
    raw = torch.randn(2, 2, 8, 10)  # tiny VAE latent (latent_channels 8 -> 2 raw channels)
    latent = patchify_latent(raw)
    assert latent.shape == (2, 8, 4, 5)
    assert torch.equal(unpatchify_latent(latent), raw)


def test_patchify_channel_correspondence():
    """Channel c0*4 + 2*i + j at (h, w) must equal raw channel c0 at (2h + i, 2w + j)."""
    b, c, h2, w2 = 1, 3, 8, 8
    raw = torch.arange(b * c * h2 * w2, dtype=torch.float32).reshape(b, c, h2, w2)
    latent = patchify_latent(raw)
    for c0 in range(c):
        for i in range(2):
            for j in range(2):
                for hp in range(0, h2 // 2):
                    for wp in range(0, w2 // 2):
                        expected = raw[0, c0, 2 * hp + i, 2 * wp + j]
                        actual = latent[0, 4 * c0 + 2 * i + j, hp, wp]
                        assert actual == expected


def test_patchify_rejects_odd_dims():
    with pytest.raises(ValueError):
        patchify_latent(torch.randn(1, 2, 3, 4))
    with pytest.raises(ValueError):
        unpatchify_latent(torch.randn(1, 6, 4, 4))


def test_bn_roundtrip_and_decode_math():
    torch.manual_seed(2)
    vae = FakeVae(latent_channels=8)
    mean, std = vae_bn_stats(vae)
    raw = torch.randn(1, 2, 8, 8)
    latent = (patchify_latent(raw) - mean) / std
    recovered = unpatchify_latent(latent * std + mean)
    assert torch.allclose(recovered, raw, rtol=0, atol=1e-5)
    # And the pipeline's own un-normalize order (scale then shift) is what decode_image mirrors.
    assert torch.allclose(latent * std + mean, patchify_latent(raw), atol=1e-5)


def test_encode_image_latent_matches_manual_math():
    torch.manual_seed(3)
    vae = FakeVae(latent_channels=8)
    image = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    latent = encode_image_latent(image, vae)
    assert latent.shape == (1, 8, 16, 16)
    mode = vae.encode(image).latent_dist.mode()
    mean, std = vae_bn_stats(vae)
    expected = (patchify_latent(mode) - mean) / std
    assert torch.allclose(latent, expected, rtol=0, atol=1e-6)


def test_decode_image_matches_pipeline_decode():
    """decode_image must reproduce MOSSImagePipeline's internal decode bit-for-bit."""
    model = tiny_transformer().eval()
    vae = FakeVae(latent_channels=8)
    pipe = MOSSImagePipeline(transformer=model, scheduler=MOSSImageScheduler(),
                             text_encoder=None, tokenizer=None, vae=vae)
    pipe.set_progress_bar_config(disable=True)
    embeds = torch.randn(1, 4, 32)
    neg = torch.randn(1, 4, 32)
    latents_in = torch.randn(1, 8, 16, 16)
    kwargs = dict(prompt_embeds=embeds, negative_prompt_embeds=neg, height=256, width=256,
                  num_inference_steps=2, guidance_scale=3.0, uncond_skip_layers=[])
    final_latents = pipe(latents=latents_in, output_type="latent", **kwargs).images
    decoded_by_pipeline = pipe(latents=latents_in, output_type="pt", **kwargs).images
    mine = decode_image(final_latents.float(), vae)[0]
    # postprocess denormalizes [-1, 1] -> [0, 1]
    assert torch.allclose(mine / 2 + 0.5, decoded_by_pipeline[0], rtol=0, atol=1e-6)


def test_decode_output_range():
    vae = FakeVae(latent_channels=8)
    latent = torch.randn(1, 8, 8, 8)
    image = decode_image(latent, vae)
    assert image.shape == (1, 3, 128, 128)  # 8x8 latent -> 16x16 raw -> x8 decode
    assert image.min() >= -1.0 and image.max() <= 1.0
