# Using the trained LoRA in ComfyUI

This page covers exporting a MOSS-LoRA-Trainer adapter for ComfyUI and wiring it
into a MOSS-Image workflow. The exported file is additive only: the trained
`adapter_model.safetensors` is read, never modified.

## 1. Why the adapter must be converted

The trainer stores keys with the module paths of the vendored transformer:

```text
layers.0.proj.q.lora_A
layers.0.proj.q.lora_B
```

ComfyUI keys the DiT under the `diffusion_model.` prefix. Its LoRA loader maps the
model weight `diffusion_model.layers.0.proj.q.weight` to the base key
`diffusion_model.layers.0.proj.q`, then looks for these three files:

```text
diffusion_model.layers.0.proj.q.lora_A.weight
diffusion_model.layers.0.proj.q.lora_B.weight
diffusion_model.layers.0.proj.q.alpha
```

The prefix and the `.weight` suffix are both required. If you load the raw
trainer adapter you get a warning for every tensor:

```text
lora key not loaded: layers.0.proj.q.lora_A
```

The converter adds `diffusion_model.`, appends `.weight` to `lora_A`/`lora_B`, and
writes one scalar `alpha` per target. It refuses unknown keys, missing A/B
pairs, rank/shape mismatches, and missing configured targets instead of
silently dropping tensors.

## 2. Export the adapter

From the trainer repository:

```bash
moss-lora-export-comfy \
  --adapter runs/my_lora \
  --output runs/my_lora/my_lora_comfy.safetensors
```

If the console script is not installed, the module entry point is equivalent:

```bash
python -m moss_lora.export_comfy \
  --adapter runs/my_lora \
  --output runs/my_lora/my_lora_comfy.safetensors
```

`--adapter` accepts either the run directory (containing `adapter_model.safetensors`
and `adapter_config.json`) or a weights file with `adapter_config.json` beside it.
Alpha and rank come from the config. The output keeps the source dtype and
values and carries the metadata:

```json
{"format": "pt", "moss_lora_comfy": "1", "base_model": "models/MOSS-Image-Base"}
```

The trained adapter (`rank=32`, `alpha=32`) exports to 756 keys:
504 LoRA weights plus 252 alpha scalars (42 layers x 6 targets x 3).

## 3. Copy the file into ComfyUI

Put `my_lora_comfy.safetensors` in the ComfyUI LoRA directory:

| Environment | Typical path |
| --- | --- |
| Windows (portable) | `ComfyUI\models\loras\my_lora_comfy.safetensors` |
| Windows (installed) | `%USERPROFILE%\Documents\ComfyUI\models\loras\my_lora_comfy.safetensors` |
| WSL2 | `/mnt/c/Users/<you>/Documents/ComfyUI/models/loras/my_lora_comfy.safetensors` |
| Linux | `<ComfyUI>/models/loras/my_lora_comfy.safetensors` |

Restart ComfyUI or refresh the node definitions so the file appears in the
`lora_name` dropdown.

## 4. Load it with `LoraLoaderModelOnly`

Use **Load LoRA (Model Only)** (`LoraLoaderModelOnly`), not the regular
`LoraLoader`. The regular node also applies the LoRA to CLIP; this adapter has
no text-encoder tensors, so that produces a second set of harmless
`lora key not loaded` warnings and changes nothing.

- `lora_name`: `my_lora_comfy.safetensors`
- `strength_model`: `0.8` to `1.0` (the workflow uses `0.9`)

The ready-to-import graph is [`workflows/moss_image_lora.json`](../workflows/moss_image_lora.json).
It is a reference graph reconstructed with the loader/save widget schemas used
by the user's workflow (three MOSS loader widgets, two seed widgets, four save
widgets). Open it from ComfyUI's workflow menu (or drag it onto the canvas),
then confirm the model path, the LoRA filename, and the prompt text. The bundled
positive/negative prompts are placeholders; paste your own caption. If your
existing workflow differs, the only required change is inserting
`LoraLoaderModelOnly` between `MOSSImageHFLoader.MODEL` and
`ModelSamplingSD3`, with the CLIP path left separate.

The MODEL chain is:

```text
MOSSImageHFLoader.MODEL
  -> LoraLoaderModelOnly
  -> ModelSamplingSD3 (shift 5.0)
  -> SkipLayerGuidanceDiTSimple (skip layer 22)
  -> KSampler
```

The CLIP path stays separate and never touches the LoRA node:

```text
MOSSImageHFLoader.CLIP -> CLIPTextEncode (positive / negative) -> KSampler
```

The VAE path used for decoding also bypasses the LoRA:

```text
MOSSImageHFLoader.VAE -> VAEDecode
```

Do not connect the LoRA node to CLIP. If you prefer to build it manually, insert
`LoraLoaderModelOnly` between the MOSS loader and `ModelSamplingSD3` exactly as
shown above.

## 5. Recommended sampling settings

| Setting | Value |
| --- | --- |
| Resolution | 1024 x 1024 or 1024 x 1536 |
| Steps | 50 |
| CFG | 3.0 |
| Sampler | `euler` |
| Scheduler | `sgm_uniform` |
| Shift (`ModelSamplingSD3`) | 5.0 |
| Skip layer guidance | layer 22 |
| `strength_model` | 0.8 - 1.0 |

The adapter was trained at 1M pixels for 5,000 steps. At 1M pixels the effect is
usually subtle; strength `0.9` is a good starting point. Lower it if colors or
textures drift, raise it only slightly if the effect is too weak. Because the
adapter targets `q`, `k`, `v`, `o`, `ffn_up_gate`, and `ffn_down` in all 42
layers with `alpha == rank`, ComfyUI's scale is exactly `1.0` at
`strength_model: 1.0`.

## 6. Troubleshooting

### `lora key not loaded`

Every tensor in the LoRA file was skipped. Check, in order:

1. You loaded `my_lora_comfy.safetensors`, not the raw
   `adapter_model.safetensors`.
2. You used `LoraLoaderModelOnly`; a regular `LoraLoader` also complains about
   the absent text-encoder keys.
3. The converted file contains `diffusion_model.layers...proj.q.lora_A.weight`
   (`python -c "from safetensors import safe_open; ..."` or simply re-run the
   export and confirm the sample keys printed by the CLI).

### Warnings from the CLIP side

If you see warnings only after connecting a CLIP output to a LoRA loader, they
come from the LoRA being checked against the text encoder. This adapter has no
text-encoder tensors. Use the model-only loader and leave the LoRA out of the
CLIP path; the warnings disappear.

### The effect is very weak

This is expected at 1M pixels after 5k steps. First confirm the LoRA is actually
applied (the log should not contain `lora key not loaded` for the converted
file), then try `strength_model` 1.0. Do not compensate by changing CFG or the
shift away from the values above unless you are deliberately comparing recipes.

## 7. Validation status

The workflow JSON is structurally validated: every link endpoint references an
existing node and slot, link types match, and the graph is acyclic. It was not
executed inside ComfyUI in this repository, because that requires the MOSS
plugin, a GPU, and the base checkpoint. Your one manual step is to copy the
exported file into `models/loras/`, open `workflows/moss_image_lora.json`, and
confirm the MOSS model and LoRA file in the two loader nodes.
