# MOSS-Image LoRA Trainer

在单卡 **RTX 5090 32GB（Blackwell，sm_120）** 上对 [OpenMOSS MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) 做 LoRA 微调：文本与 VAE 编码全部离线缓存，训练循环只保留 DiT + LoRA。

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![RTX 5090 / sm_120](https://img.shields.io/badge/RTX%205090-sm__120-76b900.svg)](docs/5090.md)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)](#状态与限制)

> [!WARNING]
> **本项目由 AI 辅助 / vibe coding 生成。** 代码、训练配方、显存估算和文档未经完整人工审计；使用前请自行审查代码，并在自己的数据上验证画质与安全性。
>
> **This project was produced with AI assistance ("vibe coding").** The code, training recipe, memory estimates and documentation have not been fully human-audited. Review the code and validate quality and safety on your own data before use. No warranty; see [LICENSE](LICENSE).

## 它是什么

本仓库是一个面向 MOSS-Image-Base 的最小 LoRA 训练器：先把图片的 VAE latent 和 caption 的 Qwen3.5 文本 embedding 离线缓存，训练时只加载冻结的 DiT 与可训练的 LoRA。它面向单张 32GB 显卡，默认 1M 像素、bf16 基座、fp32 LoRA/优化器状态，并提供采样、合并、基准测试和 ComfyUI 导出。训练配方由公开的技术报告、模型仓库和官方推理代码推导，**不是** OpenMOSS 官方训练实现。完整设计推导见 [docs/RESEARCH.md](docs/RESEARCH.md)。

## 快速开始

### 1. 安装

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

RTX 5090 需要 cu128 或更新的 wheel；驱动与验证清单见 [docs/5090.md](docs/5090.md)。

### 2. 模型

```bash
hf download OpenMOSS-Team/MOSS-Image-Base \
  --revision 1ceede80badddd6a7812069619edaa5633587192 \
  --local-dir models/MOSS-Image-Base
```

下文默认模型位于 `models/MOSS-Image-Base`；`model.path` 与所有 `--model` 参数也接受本地目录或 HF repo id。

### 3. 数据

每张图片配同名 `.txt` caption，或使用 `metadata.jsonl`（支持字符串 caption、`{long, medium, short}` 和 layout JSON）：

```text
data/
├── images/0001.jpg
├── images/0001.txt
└── images/0002.png
```

完整格式、layout、分桶与 caption 规范见 [docs/DATASET.md](docs/DATASET.md)，可直接参考 [`examples/`](examples/)。
如果 sidecar 是「tag 行 + 中/英 prose」的原始 prompt，先用
[`scripts/prepare_captions.py`](scripts/prepare_captions.py) 生成带 `artist:` 条件的
long/medium/short captions、`metadata.jsonl` 和裁剪/token 报告，见
[docs/CAPTIONS.md](docs/CAPTIONS.md)。

### 4. 缓存

```bash
moss-lora-cache --input data --output cache/1m --model models/MOSS-Image-Base \
  --target-area 1048576 --max-text-length 512 \
  --negative-prompt "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
```

缓存阶段会短暂加载 9B 文本编码器和 VAE，**不要与训练同时运行**。

### 5. 训练

```bash
moss-lora-train --config configs/train_5090_1m.yaml
moss-lora-train --config configs/train_5090_1m.yaml --resume runs/moss_1m  # 断点恢复
```

输出包括 `metrics.csv`、`samples/`、`checkpoints/` 和最终 adapter；目录结构、全部字段与调参见 [docs/CONFIG.md](docs/CONFIG.md)。

训练期间可另开一个终端监控进度（FastAPI/uvicorn 是可选依赖，默认只监听本机）：

```bash
pip install -e '.[dashboard]'
moss-lora-dashboard --root runs
```

浏览器打开 <http://127.0.0.1:7860/>；远程机器用 SSH 端口转发，详见 [docs/DASHBOARD.md](docs/DASHBOARD.md)。

### 6. 采样

```bash
moss-lora-sample --model models/MOSS-Image-Base --lora runs/moss_1m \
  --prompt "A minimalist travel poster with a cream background and a blue mountain illustration." \
  --height 1024 --width 1024 --steps 50 --cfg 3.0 --shift 5.0 --skip-layer 22 \
  --negative-prompt "nsfw, worst quality, bad anatomy, watermark, jpeg artifact" \
  --seed 42 --output out.png
```

### 7. 导出到 ComfyUI

```bash
moss-lora-export-comfy --adapter runs/moss_1m \
  --output runs/my_lora_comfy.safetensors
```

把生成的 `my_lora_comfy.safetensors` 放进 ComfyUI 的 `models/loras/`，用 `LoraLoaderModelOnly` 加载；现成工作流见 [`workflows/moss_image_lora.json`](workflows/moss_image_lora.json)，完整步骤见 [docs/COMFYUI.md](docs/COMFYUI.md)。

## 特性

- **离线缓存**：VAE latent 与文本 embedding 预计算，训练时不需要文本编码器或 VAE。
- **LoRA-only**：DiT 全程冻结，只训练注意力与 FFN 投影上的低秩 adapter。
- **省显存**：bf16 基座、fp32 LoRA/AdamW、可选 gradient checkpointing 与 SDPA 融合 attention。
- **吞吐友好**：按分辨率/aspect bucket 采样，同 batch 不 padding，自定义 collate。
- **可恢复**：精确保存/恢复模型、优化器、scheduler 与 RNG 状态。
- **可验证**：内置采样 CLI、LoRA 合并、step time/峰值显存基准。
- **实时看板**：只读监控 loss/EMA、LR、采样图、checkpoint、配置与 ETA；自动刷新，支持 SSH 端口转发。
- **ComfyUI 导出**：把训练器 key 转换为 ComfyUI 可加载的 `diffusion_model.*` 格式。
- **有实测锚点**：本地 WSL2 RTX 5090 的 20-step smoke 为 1.25 s/step、峰值 12.84GB（仅冒烟参考，见 [docs/5090.md](docs/5090.md#33-已测20-step-smoke2026-10-18)）。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/RESEARCH.md](docs/RESEARCH.md) | 架构、token 数学、flow-matching、latent/patchify、LoRA 设计、显存与吞吐分析、未知项 |
| [docs/CONFIG.md](docs/CONFIG.md) | 全部配置字段与默认值、超参建议、输出/checkpoint 布局、resume、1M/512/长跑配方 |
| [docs/DASHBOARD.md](docs/DASHBOARD.md) | 只读实时看板：安装/启动、曲线与采样、状态/ETA、HTTP API、安全与 SSH 端口转发 |
| [docs/5090.md](docs/5090.md) | 硬件基线、驱动/wheel 验证、显存规划、OOM 阶梯、性能开关、实测表 |
| [docs/DATASET.md](docs/DATASET.md) | 数据目录、metadata.jsonl、layout JSON、caption 变体与质量规范 |
| [docs/CAPTIONS.md](docs/CAPTIONS.md) | 原始 tag+prose 解析、混合画师 caption、单语言 prose、负向 tag 清理、文字/手部覆盖、分桶裁剪风险、OCR/bbox/layout 与报告字段 |
| [docs/COMFYUI.md](docs/COMFYUI.md) | adapter 转换、ComfyUI 工作流接线、采样参数与排错 |
| [docs/FAQ.md](docs/FAQ.md) | 常见问题与快速排查 |
| [SECURITY.md](SECURITY.md) | 安全与隐私报告方式 |

## 状态与限制

- OpenMOSS **没有发布官方训练代码**；本仓库配方由技术报告、`pipeline_moss_image.py` 和 User Guide 推导，属于工程选择。
- 唯一实测是 20-step、单 seed、16 张图的 smoke；**没有**长跑、多配置、batch>1、compile 或画质收敛 benchmark。
- 显存/吞吐除标注“实测”外均为解析估算，不构成性能承诺；LoRA 质量取决于数据、caption 和超参，无法保证。
- 代码与文档由 AI 辅助生成，可能存在错误；使用前请审查代码，并在自己的数据上验证输出质量与安全性。
- 本项目与 OpenMOSS 无隶属关系，也未获其背书。

## 许可与致谢

代码以 **Apache-2.0** 发布，见 [LICENSE](LICENSE)。`moss_lora/model/moss_image.py` 改编自 OpenMOSS 的推理 pipeline；来源、版本与修改范围见 [NOTICE](NOTICE)。模型与官方实现：[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)、[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)。本仓库不包含、也不分发模型权重或数据集。
