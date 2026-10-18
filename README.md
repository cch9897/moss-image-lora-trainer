# MOSS-Image LoRA Trainer

面向 **单卡 RTX 5090 32GB（Blackwell，sm_120）** 的 [OpenMOSS MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) LoRA 微调工具。

目标很明确：在 32GB 显存内跑完 1M 像素的 DiT LoRA 训练，同时保持官方推理路径的数值约定与画质。项目不追求"支持所有显卡/所有训练范式"，只把 5090 这一档硬件吃透。

> **状态与边界**
>
> OpenMOSS 目前**没有发布官方训练代码**。本仓库的训练配方由以下公开材料推导：
>
> - 模型仓库中的 `scheduler/scheduler_config.json` 与 `pipeline_moss_image.py`
> - [MOSS-Image 技术报告](https://github.com/OpenMOSS/MOSS-Image/blob/main/paper/MOSS_Image_Technical_Report.pdf)（附录 A.1/A.2，Table 5/6/7）
> - [官方 User Guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)（推理参数、LoRA 建议、caption 流程）
>
> 配方中的超参、LoRA 目标层、缓存格式是本项目的工程选择，不是官方训练配置。请在自己的数据上验证后再用于生产。文中所有显存/吞吐数字，除标注"实测"或"外部报告"外均为**估算**，不构成性能承诺。

---

## 目录

- [它是什么](#它是什么)
- [为什么针对 5090](#为什么针对-5090)
- [特性](#特性)
- [安装](#安装)
- [下载模型](#下载模型)
- [准备数据](#准备数据)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [显存与吞吐优化](#显存与吞吐优化)
- [推荐超参数](#推荐超参数)
- [采样与验证](#采样与验证)
- [常见问题](#常见问题)
- [文件结构](#文件结构)
- [许可与致谢](#许可与致谢)

---

## 它是什么

MOSS-Image-Base 是一个 5B 级别的单流 DiT 文生图模型：

| 组件 | 配置 | 训练时状态 |
| --- | --- | --- |
| DiT | 42 层，hidden 3072，24 heads（head_dim 128），FFN 8192 + SiTU gate，4D RoPE，`latent_channels=128` | 冻结基座，只训练 LoRA |
| 文本编码器 | Qwen3.5-9B，32 层，hidden 4096，输出 4096 维 | **冻结**，且训练时常驻显存的是它的**离线缓存** |
| VAE | `AutoencoderKLFlux2`，32 latent channels，8× 下采样，2×2 patch → 128 通道 | 冻结，只在缓存阶段使用 |

DiT 权重 5,082,308,608 参数，bf16 约 **10.16GB**。文本编码器权重 8,392,695,024 参数，bf16 约 **16.79GB**；VAE 约 250MB。

本项目的核心思路不是"把 9B 文本编码器量化到 4bit 塞进显存"，而是**把文本编码和 VAE 编码全部前移到离线缓存**：训练循环里只有 DiT + LoRA。这是 32GB 单卡能舒服训练 1M 像素的根本原因。

## 为什么针对 5090

- **sm_120 需要新 wheel**：RTX 50 系是 Blackwell，compute capability `(12, 0)`。旧版 PyTorch（cu118/cu126）没有 sm_120 kernel，典型报错是 `no kernel image is available for execution on the device`。必须使用 CUDA 12.8+ 的 wheel（本项目以 PyTorch 2.9 + cu128 为基准）。
- **32GB 是关键约束**：DiT bf16 10.16GB + fp32 LoRA/优化器约 1.0–1.2GB + checkpointing 后的激活约 1.5–2.5GB（1M 像素、batch 1，估算），再加 CUDA context/工作区，峰值估算约 **14–16GB**；20-step smoke 实测 **12.84GB**（compile off）。剩余显存可以用来提高 batch、开 compile、或跑验证采样。
- **算力定位要诚实**：RTX 5090 的 bf16 tensor 稠密算力（FP32 累加）为 **209.5 TFLOPS**，FP8 为 419 TFLOPS，FP4 为 1676 TFLOPS（稀疏翻倍）[NVIDIA Blackwell 架构白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)。它的 bf16 峰值低于 A100 80GB SXM（312 TFLOPS），1792GB/s 的 GDDR7 带宽也低于 A100 的 HBM（80GB SXM 为 2039GB/s）。它的优势是 32GB 显存、FP4、消费级成本和单卡可得性，而不是"bf16 绝对更快"。本项目的默认精度是 bf16，因此不要期待它无条件快过数据中心卡。
- **单卡场景**：没有 NVLink，多卡训练不在本项目范围内。

## 特性

- **两阶段离线缓存**：VAE 编码为**与官方推理完全一致**的 normalized + patchified latent（默认 fp32）；Qwen3.5 按官方 chat template 编码 caption（bf16），负向提示词也缓存。
- **训练期零文本编码器/VAE**：每步只跑 DiT 前向/反向，显存和 step time 都省下来。
- **LoRA-only**：基座全程冻结，fp32 LoRA 参数 + fp32 AdamW 状态，bf16 计算。
- **精确复现官方 latent 约定**：patchify → 128 通道 BatchNorm 归一化；采样时 denormalize → unpatchify → VAE decode。
- **官方默认采样参数**：50 steps、CFG 3.0、shift 5.0、Euler、负向分支 skip layer 22。
- **省显存训练栈**：SDPA fused attention、可选 gradient checkpointing（1M 默认开）、可选 `torch.compile(dynamic=True)`、fused AdamW。
- **吞吐友好**：按分辨率/aspect bucket 采样，同 batch 不 padding；自定义 collate + attention mask。
- **训练可控**：EMA、梯度裁剪、warmup、cosine/constant schedule、精确 resume（模型/优化器/scheduler/RNG）。
- **可验证**：`moss-lora-bench` 在目标分辨率下测 step time 与峰值显存；`moss-lora-sample` 用官方参数出图对比。

## 安装

### 前置条件

- Linux（Ubuntu 22.04/24.04 推荐）；WSL2 可用，见 [docs/5090.md](docs/5090.md)。
- Python 3.10–3.14（3.14 需 torch ≥2.9 的 cp314 wheel；本项目开发环境使用 3.14）。
- NVIDIA 驱动 ≥ 570.26（CUDA 12.8 的最低驱动要求）；`nvidia-smi` 右上角的 CUDA Version 需 ≥ 12.8。
- RTX 5090 32GB；磁盘至少 40GB（模型约 27GB + 缓存 + 输出）。

### 安装 PyTorch（sm_120）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# 必须用 cu128（或更新）的 wheel；默认 PyPI 的 torch 可能不带 sm_120
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

pip install -e .
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))   # 期望 (12, 0)
print("arch list:", torch.cuda.get_arch_list())             # 期望包含 sm_120
PY
```

期望输出中 `torch.version.cuda` 为 `12.8`，capability 为 `(12, 0)`。若 capability 正确但 arch list 没有 `sm_120`，说明 wheel 装错了。

> 如果 `pip install -e .` 试图替换已安装的 torch，先检查 `pyproject.toml` 的依赖约束；本项目要求 torch ≥ 2.9。不要在装好 cu128 wheel 后再执行 `pip install torch --upgrade`，那会从 PyPI 拉到可能不匹配的构建。

## 下载模型

```bash
# huggingface_hub >= 0.34 提供 hf 命令
hf download OpenMOSS-Team/MOSS-Image-Base \
  --revision 1ceede80badddd6a7812069619edaa5633587192 \
  --local-dir models/MOSS-Image-Base
```

配置文件默认也固定到该 commit（`model.revision`）。旧版本 CLI 使用 `huggingface-cli download ... --revision ... --local-dir ...`。

完整仓库约 **27GB**（DiT 10.16GB + 文本编码器 16.79GB + VAE 0.25GB + 配置/代码）。国内网络可使用 `HF_ENDPOINT=https://hf-mirror.com`，或从 ModelScope 下载后转换为相同目录结构。

**使用本地目录**：`model.path`（以及所有命令的 `--model`）既接受 HF repo id，也接受本地目录：

```yaml
model:
  path: models/MOSS-Image-Base
  local_files_only: true
```

本地目录只需包含代码实际加载的组件：

```text
models/MOSS-Image-Base/
├── transformer/              # DiT 权重 + config（训练必需）
├── text_encoder/             # Qwen3.5-9B 权重 + config（缓存/采样需要）
├── tokenizer/                # tokenizer.json + chat_template.jinja（缓存/采样需要）
└── vae/                      # AutoencoderKLFlux2 权重 + config（缓存/PNG 采样需要）
```

缺少 `tokenizer/` 或 `chat_template.jinja` 会改变文本条件。代码**不读取**模型目录里的 `scheduler/`、`model_index.json` 或 `pipeline_moss_image.py`：scheduler 与 pipeline 使用仓库内 vendored 的 `moss_lora.model.moss_image`。缓存与 PNG 采样需要文本编码器和 VAE；纯 latent 训练（以及没有 VAE 时的 latent 验证）只需要 `transformer/`。

## 准备数据

支持两种最简形式：

```text
data/
├── images/
│   ├── 0001.jpg
│   ├── 0001.txt
│   ├── 0002.png
│   └── 0002.txt
└── metadata.jsonl          # 可选；存在时优先于 .txt
```

`metadata.jsonl` 每行一张图：

```jsonl
{"file_name": "images/0001.jpg", "caption": "A detailed English caption..."}
{"file_name": "images/0002.png", "caption": {"long": "...", "medium": "...", "short": "..."}}
{"file_name": "images/0003.png", "layout": {"summary": {"meta": ["poster"], "canvas": "single", "caption": "..."}, "items": [{"label": "text", "caption": "...", "text": "EXPLORE", "bbox": [-416, -384, -288, 384]}], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}}
```

- `caption` 为字符串：单一 caption。
- `caption` 为 `{long, medium, short}`：三个变体都缓存，训练时等概率采样（对齐技术报告的 1:1:1 recaption 混合）。
- `layout`：官方 layout JSON，训练时作为紧凑 JSON 字符串送入文本编码器。

完整规则、分辨率/分桶、caption 质量要求、官方 caption prompt 见 **[docs/DATASET.md](docs/DATASET.md)**。示例文件见 [`examples/`](examples/)。

## 快速开始

以下命令假设模型已下载到 `models/MOSS-Image-Base`，缓存目录为 `cache/1m`，输出目录为 `runs/moss_1m`。

### 1. 缓存 latent 与文本 embedding

```bash
moss-lora-cache \
  --input data \
  --output cache/1m \
  --model models/MOSS-Image-Base \
  --target-area 1048576 \
  --max-text-length 512 \
  --latent-dtype fp32 \
  --negative-prompt "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
```

- `--input` 接受目录或单个 `.jsonl`；`--output` 是缓存目录（`index.json` + `item_%05d.safetensors` + `negative.safetensors`）。
- `--target-area 1048576` 即 1M 像素（1024×1024 附近）。bucket 边长统一对齐到 64（满足模型对 16 的倍数要求），范围约 512–2048。
- `--max-text-length 512` 是 caption token 截断上限（CLI 默认 1024；示例用 512 控制 attention 成本）。实际存储的是真实序列长度，不会 padding。官方推理支持 4096；layout JSON 等长文本可提高到 1024–2048。
- `--latent-dtype fp32` 是默认值；`bf16` 可把缓存减半，但会损失一点 latent 精度，质量敏感场景建议保持 fp32。
- 负向提示词只编码一次并缓存，训练与验证采样共用。
- `--missing-caption skip`（默认）会带图片路径告警并跳过缺失/空 caption，继续处理其余数据；`--missing-caption error` 遇到第一张坏图即报错。请优先补齐 caption，不要依赖跳过，空 caption 会训练出无条件生成行为。

512 数据集把面积改成 262144：

```bash
moss-lora-cache --input data --output cache/512 --model models/MOSS-Image-Base \
  --target-area 262144 --max-text-length 512
```

`--output` 必须与配置里的 `data.cache_dir` 一致（1M 用 `cache/1m`，512 用 `cache/512`）。

缓存阶段会依次/分时占用 GPU 加载 9B 文本编码器（16.79GB）和 VAE（fp32），**不要与训练同时跑**。峰值显存估算约 18–20GB（TE 权重 + 激活 + context）；32GB 足够，但如果同时开浏览器/其他训练仍可能紧张。缓存是幂等的：重复执行会复用已存在的有效条目。

### 2. 训练

```bash
moss-lora-train --config configs/train_5090_1m.yaml
```

从 checkpoint 恢复（传 run 目录会自动选最新的 `step_*`）：

```bash
moss-lora-train --config configs/train_5090_1m.yaml --resume runs/moss_1m
```

常用 CLI 覆盖（不必改 YAML）：

```bash
moss-lora-train --config configs/train_5090_1m.yaml \
  --max-steps 200 --output-dir runs/smoke --batch-size 1 \
  --device cuda --val-ids item_00000,item_00003,item_00007
```

输出目录包含 `metrics.csv`、`samples/step_%06d/`、`checkpoints/step_%06d/`；每个 checkpoint 有 `adapter_model.safetensors`、`adapter_config.json`（可选 `adapter_ema.safetensors`）、`optimizer.pt`、`lr_scheduler.pt`、`rng.pt`、`trainer_state.json`。最终 adapter 也会复制到输出目录根。

### 3. 采样验证

```bash
moss-lora-sample \
  --model models/MOSS-Image-Base \
  --lora runs/moss_1m \
  --prompt "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top." \
  --height 1024 --width 1024 \
  --steps 50 --cfg 3.0 --shift 5.0 --skip-layer 22 \
  --negative-prompt "nsfw, worst quality, bad anatomy, watermark, jpeg artifact" \
  --seed 42 \
  --output out.png
```

加 `--ema` 使用 EMA 权重（checkpoint/run 根中存在 `adapter_ema.safetensors` 时）。`--lora` 指向包含 `adapter_model.safetensors` 与 `adapter_config.json` 的目录（run 根或某个 `checkpoints/step_*`）。`--dtype` 控制 DiT 精度（默认 `bf16`）；`--skip-layer -1` 关闭 skip；`--output-type latent` 只输出 latent，用于没有 VAE 的冒烟测试；`--max-text-length` 默认 1024（与缓存默认一致），layout 等长文本可显式提高到 2048–4096。`--transformer <dir>` 可覆盖 DiT 路径（其余组件仍从 `--model` 加载），用于加载下面合并后的目录。

### 4. 合并 LoRA

```bash
moss-lora-merge \
  --model models/MOSS-Image-Base \
  --lora runs/moss_1m \
  --output merged/
```

合并后得到的是**只含 transformer** 的 `save_pretrained` 目录，部署时不再需要加载 LoRA。用本仓库的采样 CLI 验证时，把 base 模型继续作为 `--model`（提供文本编码器、tokenizer 与 VAE），并用 `--transformer` 指向合并目录：

```bash
moss-lora-sample \
  --model models/MOSS-Image-Base \
  --transformer merged/ \
  --prompt "A minimalist travel poster ..." \
  --output merged_check.png
```

官方 `MOSSImagePipeline` 也可将 `merged/` 作为 transformer 组件加载；`merged/` 本身不是完整的模型根目录。

想在 ComfyUI 里直接使用训练出的 LoRA？训练器 adapter 的 key 缺少 ComfyUI 需要的
`diffusion_model.` 前缀和 `.weight` 后缀，先运行 `moss-lora-export-comfy` 转换，再使用
`LoraLoaderModelOnly`（不要用带 CLIP 的 `LoraLoader`）。完整步骤见
[docs/COMFYUI.md](docs/COMFYUI.md)，现成工作流在
[workflows/moss_image_lora.json](workflows/moss_image_lora.json)。

### 5. 基准测试

```bash
# 默认：1024×1024，batch 1，10 步
moss-lora-bench --model models/MOSS-Image-Base \
  --height 1024 --width 1024 --batch-size 1 --steps 10 --warmup 3

# 开启 compile
moss-lora-bench ... --compile

# 关闭 gradient checkpointing（显存换速度）
moss-lora-bench ... --no-checkpointing

# 机器可读输出
moss-lora-bench ... --json
```

建议同一配置连续跑两次，取第二次的稳定值；第一次包含编译/warmup。把结果填进 [docs/5090.md](docs/5090.md) 的测量表。

> 已做一次 **20-step 真实模型 smoke**（本地 WSL2，RTX 5090 32GB，`torch 2.14.1+cu130`，16 张 1M 图，rank 32，batch 1，checkpointing on，compile off）：steady **1.25 s/step**（首次 2.59s）、峰值 **12.84GB**、loss mean 0.423、训练墙钟 46.6s。这是冒烟数据，**不是**长跑或多配置 benchmark；没有测 compile、batch>1、长步数或画质收敛。其余显存/速度为解析估算。WSL2 跑 GPU 前需 `ulimit -n 65536`，原因见 [docs/5090.md](docs/5090.md#7-windows--wsl2-补充)。

### 冒烟测试（CPU）

先合成极小模型/缓存，再跑训练：

```bash
python scripts/make_tiny_fixtures.py
moss-lora-train --config configs/tiny_cpu.yaml
```

`tiny_cpu.yaml` 用极小规模数据验证入口、配置解析和训练循环，不需要 9B 编码器，也不产生可用模型。它只回答"代码能否跑通"。该配置启用了验证采样但没有 VAE，因此 `samples/step_*/` 下写入的是 `item_*.latent.pt`（latent-only validation），而不是 PNG。无法安装 CLI 时可用 `python -m moss_lora.trainer --config configs/tiny_cpu.yaml` 等价运行。

## 配置说明

下面是 `configs/train_5090_1m.yaml` 的等价结构（字段以仓库中的 YAML 为准）：

```yaml
model:
  path: models/MOSS-Image-Base
  revision: 1ceede80badddd6a7812069619edaa5633587192
  dtype: bf16
  local_files_only: true

data:
  cache_dir: cache/1m
  num_workers: 4
  batch_size: 1
  drop_last: true
  pin_memory: true

train:
  device: cuda
  output_dir: runs/moss_1m
  max_steps: 2000
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
  warmup_steps: 50
  lr_scheduler: cosine
  max_grad_norm: 1.0
  seed: 42
  gradient_checkpointing: true
  compile: false
  compile_mode: default
  sdpa_backend: auto
  ema_decay: 0.0
  timestep_sampling: logit_normal
  logit_normal_mean: -1.09
  logit_normal_std: 0.95
  sigma_cutoff: 0.95
  uniform_mix: 0.01

lora:
  rank: 32
  alpha: 32
  dropout: 0.0
  targets: [q, k, v, o, ffn_up_gate, ffn_down]
  text_projection: false
  final_linear: false

logging:
  log_every: 10
  save_every: 200
  sample_every: 250
  sample_steps: 30
  sample_cfg: 3.0
  sample_skip_layer: 22
  sample_negative_prompt: "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
  sample_val_ids: []
```

### 关键字段

| 字段 | 典型值 | 说明与调参建议 |
| --- | --- | --- |
| `model.path` | HF id 或本地目录 | 指向包含 `transformer/` 的模型根目录 |
| `model.revision` | 固定 commit | 默认 `1ceede80badddd6a7812069619edaa5633587192`；Hub id 用于固定版本；本地目录加载时该字段不生效（不会切换文件版本） |
| `model.dtype` | `bf16` | 基座计算精度，可选 `fp32`（调试）；不要用 fp16（range 不够） |
| `model.local_files_only` | `true`（本地目录） | 强制离线加载；Hub id 场景设为 `false` |
| `data.cache_dir` | `cache/1m` | 512 配置为 `cache/512`；改分辨率/`target-area` 后必须重建 |
| `data.batch_size` | 1（1M） / 4（512） | 1M 下 32GB 建议 1–2；512 下 4–8 |
| `data.num_workers` | 4 | 4–8；缓存已在磁盘，worker 只做读取/collate |
| `data.drop_last` | `true` | 固定 `max_steps` 时避免最后一个小 batch |
| `data.pin_memory` | `true` | 配合非阻塞拷贝 |
| `train.device` | `cuda` | `cpu` 仅用于 tiny 冒烟测试 |
| `train.output_dir` | `runs/moss_1m` | run 目录：metrics/samples/checkpoints；512 配置为 `runs/moss_512` |
| `train.max_steps` | 2000（1M） / 1500（512） | 优化器步数，不含 gradient accumulation 的微步 |
| `train.gradient_accumulation_steps` | 1 | 显存换有效 batch；梯度按累积步数缩放 |
| `train.learning_rate` | `1e-4` | 建议 5e-5–2e-4；数据越少/rank 越大，越靠下限 |
| `train.weight_decay` | `0.01` | 只作用于 LoRA 权重，norm/bias 不衰减 |
| `train.betas` | `[0.9, 0.95]` | 对齐技术报告 Table 7；AdamW ε=1e-8 |
| `train.warmup_steps` | 50 | 步数少时 50–100 足够 |
| `train.lr_scheduler` | `cosine` | 可选 `constant`；LoRA 短跑常用后者 |
| `train.max_grad_norm` | `1.0` | 对齐报告；出现尖峰时不要盲目调大 |
| `train.gradient_checkpointing` | `true` | 1M 必开；512 在大 batch 时也建议开 |
| `train.compile` | `false` | 首次编译慢；稳定形状后再开，见 [docs/5090.md](docs/5090.md) |
| `train.compile_mode` | `default` | `reduce-overhead` 会用 CUDA Graphs，显存更高、对动态形状更敏感 |
| `train.sdpa_backend` | `auto` | 可强制 `flash` / `efficient` / `cudnn` / `math` 做对比 |
| `train.ema_decay` | `0.0` | 0=关闭；开启后额外保存一份 LoRA 权重；0.999 适合 1k+ 步 |
| `train.timestep_sampling` | `logit_normal` | 默认对齐报告最终阶段；`uniform_grid` 仅用于消融/调试 |
| `train.logit_normal_mean` | `-1.09` | 截断 logit-normal 的 μ |
| `train.logit_normal_std` | `0.95` | 截断 logit-normal 的 s |
| `train.sigma_cutoff` | `0.95` | 裁剪阈值，之后重标到 1000 步网格 |
| `train.uniform_mix` | `0.01` | 1% 离散均匀混合，保证低噪声端覆盖 |
| `lora.rank` | `32` | 16/32/64 的取舍见下文 |
| `lora.alpha` | 等于 rank | `null` 时自动取 rank；alpha/rank 决定缩放 |
| `lora.dropout` | `0.0` | 小数据过拟合时可试 0.05 |
| `lora.targets` | `[q,k,v,o,ffn_up_gate,ffn_down]` | 每个 DiT block 的注意力与 FFN 投影 |
| `lora.text_projection` | `false` | 是否给 text adapter 加 LoRA；默认不动，优先保持文本条件稳定 |
| `lora.final_linear` | `false` | 是否给 `final_linear_video` 加 LoRA；默认关闭，避免速度尺度漂移 |
| `logging.log_every` | 10 | 控制台/CSV 日志间隔 |
| `logging.save_every` | 200 | 与 `max_steps` 匹配；磁盘占用约每份 LoRA 0.3GB（rank 32） |
| `logging.sample_every` | 250 | 用缓存的验证 prompt embedding 出图；建议 250–1000；`0` 关闭 |
| `logging.sample_steps` | 30 | 30 是省时间的默认；与官方 50 步对比画质时显式设为 50 |
| `logging.sample_cfg` | 3.0 | 官方默认；短 prompt 可升到 4–6 |
| `logging.sample_skip_layer` | 22 | 官方推荐；负向分支跳过该层；`-1` 关闭 |
| `logging.sample_negative_prompt` | 官方负向 | 与缓存时的 `--negative-prompt` 保持一致 |
| `logging.sample_val_ids` | `[]` | 固定验证样本 id（见缓存的 `index.json`）；为空时即使 `sample_every > 0` 也不采样、不加载 VAE |

三份配置：

| 配置 | 用途 | 关键差异 |
| --- | --- | --- |
| `configs/train_5090_1m.yaml` | 1M 像素主配置 | batch 1，rank 32，lr 1e-4，checkpointing 开，2000 步 |
| `configs/train_5090_512.yaml` | 512 像素快速迭代 | batch 4，1500 步，checkpointing 开；显存足够可关（见 YAML） |
| `configs/tiny_cpu.yaml` | CPU 冒烟测试 | 极小模型/数据，仅验证流程 |

## 显存与吞吐优化

| 手段 | 作用 | 代价/限制 | 建议 |
| --- | --- | --- | --- |
| 离线缓存 latent + text embedding | 训练时移除约 16.79GB 文本编码器与 VAE；每步不再做文本/VAE 前向 | 一次性预处理；缓存占磁盘（1M 像素约 2MB latent/图 + 文本 1–4MB/图） | 默认开启，本项目核心 |
| bf16 基座 | 权重 10.16GB；与官方推理精度一致 | 不能用 fp16 替代 | 默认 |
| fp32 LoRA + fp32 AdamW | 小参数量的数值稳定 | 约 1.0–1.2GB（rank 32） | 默认；不要用 bf16 优化器状态 |
| SDPA fused attention | 避免物化 S×S attention 矩阵；Blackwell 上有融合 kernel | 极端 mask/形状可能退回 math | `auto`；必要时显式测 `flash`/`cudnn` |
| Gradient checkpointing | 激活从约 20GB 降到约 1–2GB（1M，估算） | 多一次前向，计算约 +33% | 1M 默认开；512 大 batch 开 |
| `torch.compile(dynamic=True)` | 融合 pointwise/减少 launch 开销，常见 10–30% 提升（估算） | 首次编译数十秒到数分钟；动态形状易重编译 | 形状稳定后再开；先跑 bench |
| fused AdamW | 单 kernel 更新，减少 optimizer 开销 | 仅 CUDA；对显存无影响 | 默认可用则用 |
| Bucket sampler + 无 padding collate | 同 batch 同形状，避免 padding token 浪费 | 需要足够多同 bucket 样本 | 默认；数据 aspect 多样时收益明显 |
| 采样阶段 TE offload | 避免 TE 16.79GB + DiT 10.16GB + VAE 同时驻留（>27GB） | 采样前多一次 CPU↔GPU 搬运 | 采样/验证默认 |
| EMA | 权重更平滑，通常更稳 | 额外 0.3GB（rank 32）+ 每步更新 | 1k 步以上可开 |

显存估算（**估算**，1M = 1024×1024，rank 32，bf16 基座）：

| 场景 | 固定占用 | 激活 | 峰值 | 结论 |
| --- | --- | --- | --- | --- |
| 1M, batch 1, checkpointing | ~11.4GB | ~1.5–2.5GB | **~14–16GB** | 推荐起点 |
| 1M, batch 1, 无 checkpointing | ~11.4GB | ~20–23GB | ~33–36GB | 32GB 下大概率 OOM |
| 1M, batch 2, checkpointing | ~11.4GB | ~3–4GB | ~16–18GB | 可尝试；注意碎片 |
| 512, batch 4, checkpointing | ~11.4GB | ~2–3GB | ~15–17GB | 快速迭代 |
| 512, batch 4, 无 checkpointing | ~11.4GB | ~26–28GB | ~39–41GB | OOM |
| 1536, batch 1, checkpointing | ~11.4GB | ~5–7GB | ~18–21GB | 可行但更慢 |

> 口径：权重按 safetensors 的十进制 GB；显存峰值为 GiB 级估算，含 CUDA context 与 allocator 余量。实际值受驱动、compile、attention backend、文本长度影响。以 `moss-lora-bench` 的输出为准。

## 推荐超参数

### LoRA rank

| rank | 可训练参数（42 blocks，含 fused `ffn_up_gate`） | fp32 参数+梯度+AdamW（估算） | 适用 |
| --- | --- | --- | --- |
| 16 | 37.2M | ~0.59GB | 风格、构图、轻量域适配 |
| 32 | 74.3M | ~1.19GB | 默认；人物/产品/常见风格 |
| 64 | 148.6M | ~2.38GB | 复杂概念、文字排版、较大数据集 |

参数按 `r × (in + out)` 计算；`ffn_up_gate` 是 fused 3072→16384 层。若按 8192 输出维度估算（漏掉 gate 的另一半），rank 32 会得到 63.3M；按官方 fused 层实际是 74.3M——**以训练启动日志打印的可训练参数为准**。alpha 通常取 rank；dropout 默认 0。

### 学习率与步数

- **lr**：5e-5 – 2e-4。默认 1e-4。小数据集（<100 图）、rank 64 建议 5e-5 – 1e-4；大数据集可到 2e-4。
- **steps**：1M、batch 1 下 1–3k。20–100 张图的风格 LoRA 约 500–1500 步；500+ 张图的人物/域适配约 2–5k 步；1–5k 张图的稳健概念可到 3–10k 步。更准确的算法是"目标 epoch 数 × 图片数 / batch size"，例如 200 张图、batch 1、目标 10 epoch ≈ 2000 步。
- **batch**：1M 下 1–4；512 下 4–8。显存不够时优先 gradient accumulation，而不是硬抬 batch。
- **warmup**：50–200 步。
- **caption 质量**：比 rank 和 lr 更影响最终效果。文字、排版、构图必须描述准确；详见 [docs/DATASET.md](docs/DATASET.md)。

### 训练稳定性

- 损失是 velocity 的 fp32 MSE，**没有 sigma/SNR 加权**；不同 sigma 采样下 loss 的绝对值不可直接跨配置比较。
- 梯度裁剪 1.0；出现 NaN 先降 lr、检查缓存是否损坏（latent NaN）、再检查 caption 是否为空。
- EMA 只在步数足够时有用；短跑（<500 步）收益有限。

## 采样与验证

MOSS-Image-Base 是 base 模型，没有做 CFG/步数蒸馏，**推理参数必须按官方推荐设置**，否则容易得到过饱和、细节崩坏或文字错误的结果。

| 参数 | 官方默认 | 说明 |
| --- | --- | --- |
| steps | ≥ 50 | 低于 50 步质量下降明显 |
| CFG | 3.0 | 推荐范围 3.0–6.0；长 prompt 用 3.0，短 prompt 可提高 |
| shift | 5.0 | scheduler 配置默认值 |
| sampler | Euler | 官方推荐 |
| skip layer | 22（负向分支） | 官方 `uncond skip` 推荐；22/24/26 都可试 |
| negative prompt | `nsfw, worst quality, bad anatomy, watermark, jpeg artifact` | 官方测试有效 |
| 分辨率 | 1M / 1.5M / 2M 像素 | 边长必须是 16 的倍数；0.25M–4M 可外推 |

验证协议建议：

1. 固定 4–8 个 holdout prompt 和 seed，训练前先出基线图。
2. 每 250–500 步用**完全相同**的 steps/CFG/skip layer/分辨率出图。
3. 同时看 raw 与 EMA（`--ema`），不要只挑最好看的一张。
4. 检查：文字是否逐字正确、layout 是否遵守、风格是否偏移、是否出现过度锐化/塑料感。
5. 训练 loss 下降但 holdout 变差，通常是过拟合或 caption 过拟合到固定句式；降低步数/rank/lr，或补充多样化 caption。

## 常见问题

| 问题 | 快速处理 |
| --- | --- |
| 1M 训练 OOM | 确认 checkpointing 已开、batch=1、没有残留 TE；设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；见 [docs/5090.md](docs/5090.md) |
| `no kernel image is available` | 装错 wheel；必须 cu128+，capability `(12,0)` |
| dtype 报错 / 出图发灰 | 基座用 bf16，不要用 fp16；LoRA/优化器状态保持 fp32；VAE decode 用 fp32 |
| 出图质量差/过饱和 | 用 50 步、CFG 3.0、shift 5.0、skip layer 22；确认采样脚本没有改这些默认 |
| LoRA 效果太强/太弱 | 调 lr/步数/rank；alpha=rank；部署时可调 LoRA scale 或直接 merge |
| 缓存占太多磁盘 | 1M 约 2MB latent/图；可用 512 缓存做迭代，最终再重建 1M |
| ComfyUI 报 `lora key not loaded` | 训练器 adapter 的 key 是 `layers.*.proj.*.lora_A`，ComfyUI 需要 `diffusion_model.*.lora_A.weight` 和 `.alpha`；先用 `moss-lora-export-comfy` 转换，并用 `LoraLoaderModelOnly`，见 [docs/COMFYUI.md](docs/COMFYUI.md) |
| 改分辨率后报错/效果差 | latent 与分辨率绑定；改 `target-area` 后必须重建缓存 |
| 断点恢复 | `--resume <run_dir 或 checkpoint_dir>`；run_dir 自动选最新 `step_*`；保持同一 config 与数据顺序 |
| Windows | 建议 WSL2；原生 Windows 未验证，见 [docs/FAQ.md](docs/FAQ.md) |
| 多卡 | 不支持；单卡 5090 是本项目目标 |

更多见 **[docs/FAQ.md](docs/FAQ.md)**。

## 文件结构

```text
MOSS-Lora-Trainer/
├── configs/
│   ├── train_5090_1m.yaml
│   ├── train_5090_512.yaml
│   └── tiny_cpu.yaml
├── docs/
│   ├── 5090.md
│   ├── COMFYUI.md
│   ├── DATASET.md
│   ├── FAQ.md
│   ├── RESEARCH.md
│   └── caption_prompt.txt
├── examples/
│   ├── layout.example.json
│   └── metadata.example.jsonl
├── moss_lora/
│   ├── model/moss_image.py # vendored 官方 pipeline（含 checkpointing）
│   ├── lora.py             # LoRALinear / adapter 存取
│   ├── cache.py            # latent + text 缓存
│   ├── dataset.py          # bucket sampler + collate
│   ├── text.py             # Qwen3.5 chat template 编码
│   ├── trainer.py          # 训练循环 / resume / EMA
│   ├── sample.py           # TE offload + DiT/VAE 采样
│   ├── merge.py            # LoRA 合并
│   ├── export_comfy.py     # 导出 ComfyUI 兼容 LoRA
│   ├── benchmark.py        # step time / 峰值显存
│   ├── config.py
│   └── utils.py
├── workflows/
│   └── moss_image_lora.json # ComfyUI 工作流（LoraLoaderModelOnly）
├── scripts/
│   └── make_tiny_fixtures.py # 生成 CPU 冒烟用的极小模型/缓存
├── tests/
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── NOTICE
├── SECURITY.md
└── README.md
```

命令入口：`moss-lora-cache`、`moss-lora-train`、`moss-lora-sample`、`moss-lora-merge`、`moss-lora-bench`、`moss-lora-export-comfy`。所有命令支持 `--help`。

## 许可与致谢

- 本仓库代码：**Apache-2.0**。
- 模型与官方推理代码：[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)、[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)，Apache-2.0。本项目的 latent/patchify/采样约定直接对齐其 `pipeline_moss_image.py`。
- 文本编码器 Qwen3.5-9B、VAE（FLUX.2）等组件遵循各自上游许可；商用前请分别确认。
- 训练配方参考了 MOSS-Image 技术报告附录 A.2（噪声采样 Table 6、优化器 Table 7）与官方 User Guide 的 caption 流程。第三方社区训练器 [mrfakename/moss-image-trainer](https://huggingface.co/spaces/mrfakename/moss-image-trainer) 提供了一个可对照的实现，本项目在文档中仅将其 A100 数据作为**外部参考**引用。
