# 配置、训练运行与超参数

本文是完整配置参考：字段含义、默认值、输出布局、断点恢复和超参建议。架构与数学推导见 [RESEARCH.md](RESEARCH.md)，硬件/显存规划见 [5090.md](5090.md)，数据格式见 [DATASET.md](DATASET.md)。

---

## 1. 配置文件

配置为 YAML，分 `model` / `data` / `train` / `lora` / `logging` 五个 section。未知字段会直接报错，避免拼写错误被静默忽略。仓库自带：

| 配置 | 用途 | 关键差异 |
| --- | --- | --- |
| [`configs/train_5090_1m.yaml`](../configs/train_5090_1m.yaml) | 1M 像素主配置 | batch 1，rank 32，lr 1e-4，checkpointing 开，2000 步 |
| [`configs/train_5090_512.yaml`](../configs/train_5090_512.yaml) | 512 像素快速迭代 | batch 4，1500 步，checkpointing 开；显存足够可关 |
| [`configs/tiny_cpu.yaml`](../configs/tiny_cpu.yaml) | CPU 冒烟测试 | 极小模型/数据，仅验证入口与训练循环 |
| 长跑参考（未随仓库提供） | 1M 长训练 | batch 1、rank 32、5000 步；`log_every: 20`、`save_every: 500`、`sample_every: 1000` |

长跑参考只是把 1M 配置的步数与日志间隔放大，不包含任何数据集或路径；按自己的数据规模调整步数即可。步数估算见 §5.2。

## 2. 完整配置样例（1M）

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

## 3. 字段说明

| 字段 | 典型值 | 说明与调参建议 |
| --- | --- | --- |
| `model.path` | HF id 或本地目录 | 指向包含 `transformer/` 的模型根目录 |
| `model.revision` | 固定 commit | 默认 `1ceede80badddd6a7812069619edaa5633587192`；本地目录加载时该字段不生效 |
| `model.dtype` | `bf16` | 基座计算精度；`fp32` 仅用于调试，不要用 fp16（range 不够） |
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
| `train.compile` | `false` | 首次编译慢；稳定形状后再开，见 [5090.md](5090.md) |
| `train.compile_mode` | `default` | `reduce-overhead` 用 CUDA Graphs，显存更高、对动态形状更敏感 |
| `train.sdpa_backend` | `auto` | 可强制 `flash` / `efficient` / `cudnn` / `math` 做对比 |
| `train.ema_decay` | `0.0` | 0=关闭；开启后额外保存一份 LoRA 权重；0.999 适合 1k+ 步 |
| `train.timestep_sampling` | `logit_normal` | 默认对齐报告最终阶段；`uniform_grid` 仅用于消融/调试 |
| `train.logit_normal_mean` | `-1.09` | 截断 logit-normal 的 μ |
| `train.logit_normal_std` | `0.95` | 截断 logit-normal 的 s |
| `train.sigma_cutoff` | `0.95` | 裁剪阈值，之后重标到 1000 步网格 |
| `train.uniform_mix` | `0.01` | 1% 离散均匀混合，保证低噪声端覆盖 |
| `lora.rank` | `32` | 16/32/64 的取舍见 §5.1 |
| `lora.alpha` | 等于 rank | `null` 时自动取 rank；alpha/rank 决定缩放 |
| `lora.dropout` | `0.0` | 小数据过拟合时可试 0.05 |
| `lora.targets` | `[q,k,v,o,ffn_up_gate,ffn_down]` | 每个 DiT block 的注意力与 FFN 投影 |
| `lora.text_projection` | `false` | 是否给 text adapter 加 LoRA；默认不动，优先保持文本条件稳定 |
| `lora.final_linear` | `false` | 是否给 `final_linear_video` 加 LoRA；默认关闭，避免速度尺度漂移 |
| `logging.log_every` | 10 | 控制台/CSV 日志间隔 |
| `logging.save_every` | 200 | 与 `max_steps` 匹配；每份 rank-32 LoRA 约 0.3GB |
| `logging.sample_every` | 250 | 用缓存的验证 prompt embedding 出图；建议 250–1000；`0` 关闭 |
| `logging.sample_steps` | 30 | 省时默认；与官方 50 步对比画质时显式设为 50 |
| `logging.sample_cfg` | 3.0 | 官方默认；短 prompt 可升到 4–6 |
| `logging.sample_skip_layer` | 22 | 官方推荐；负向分支跳过该层；`-1` 关闭 |
| `logging.sample_negative_prompt` | 官方负向 | 与缓存时的 `--negative-prompt` 保持一致 |
| `logging.sample_val_ids` | `[]` | 固定验证样本 id（见缓存 `index.json`）；为空时即使 `sample_every > 0` 也不采样、不加载 VAE |

## 4. 输出目录、checkpoint 与恢复

一次训练在 `train.output_dir` 下生成：

```text
runs/moss_1m/
├── metrics.csv
├── adapter_model.safetensors      # 最终 adapter
├── adapter_config.json
├── adapter_ema.safetensors        # 仅 EMA 开启时
├── samples/step_%06d/             # 验证采样；无 VAE 时写 *.latent.pt
└── checkpoints/step_%06d/
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── adapter_ema.safetensors    # 可选
    ├── optimizer.pt
    ├── lr_scheduler.pt
    ├── rng.pt
    └── trainer_state.json
```

`metrics.csv` 按 `log_every` 追加 step、loss、lr、step time 等字段；`trainer_state.json` 记录全局步数与数据顺序状态。

训练期间可以启动只读看板，观察 loss/EMA、学习率、采样图、checkpoint 与 ETA（不会写入 run，也不会提供权重下载）：

```bash
pip install -e '.[dashboard]'
moss-lora-dashboard --root runs --port 7860
```

看板优先读取 run 根目录的 `trainer_state.json`，不存在时自动回退到最新 checkpoint 内的副本；`adapter_config.json` 同理。完整用法、API、安全边界与 SSH 端口转发见 [DASHBOARD.md](DASHBOARD.md)。

恢复训练：

```bash
moss-lora-train --config configs/train_5090_1m.yaml --resume runs/moss_1m
```

`--resume` 接受 run 目录（自动选最新 `step_*`）或具体 checkpoint 目录。恢复会还原模型、优化器、scheduler 与 RNG；请保持同一配置、同一缓存和相同数据顺序，否则“精确恢复”只剩权重恢复的意义。

常用 CLI 覆盖（不必改 YAML）：

```bash
moss-lora-train --config configs/train_5090_1m.yaml \
  --max-steps 200 --output-dir runs/smoke --batch-size 1 \
  --device cuda --val-ids item_00000,item_00003,item_00007
```

## 5. 超参数建议

### 5.1 LoRA rank

| rank | 可训练参数（42 blocks，含 fused `ffn_up_gate`） | fp32 参数+梯度+AdamW（估算） | 适用 |
| --- | --- | --- | --- |
| 16 | 37.2M | ~0.59GB | 风格、构图、轻量域适配 |
| 32 | 74.3M | ~1.19GB | 默认；人物/产品/常见风格 |
| 64 | 148.6M | ~2.38GB | 复杂概念、文字排版、较大数据集 |

参数按 `r × (in + out)` 计算；`ffn_up_gate` 是 fused 3072→16384 层。若按 8192 输出维度估算（漏掉 gate 的另一半），rank 32 会得到 63.3M；按官方 fused 层实际是 74.3M——**以训练启动日志打印的可训练参数为准**。alpha 通常取 rank；dropout 默认 0。

### 5.2 学习率与步数

- **lr**：5e-5 – 2e-4，默认 1e-4。小数据集（<100 图）、rank 64 建议 5e-5 – 1e-4；大数据集可到 2e-4。
- **steps**：1M、batch 1 下 1–3k。20–100 张图的风格 LoRA 约 500–1500 步；500+ 张图的人物/域适配约 2–5k 步；1–5k 张图的稳健概念可到 3–10k 步。更准确的算法是“目标 epoch 数 × 图片数 / batch size”，例如 200 张图、batch 1、目标 10 epoch ≈ 2000 步。
- **batch**：1M 下 1–4；512 下 4–8。显存不够时优先 gradient accumulation，而不是硬抬 batch。
- **warmup**：50–200 步。
- **caption 质量**：比 rank 和 lr 更影响最终效果。文字、排版、构图必须描述准确；详见 [DATASET.md](DATASET.md)。

### 5.3 训练稳定性

- 损失是 velocity 的 fp32 MSE，**没有 sigma/SNR 加权**；不同 sigma 采样下 loss 的绝对值不可直接跨配置比较。
- 梯度裁剪 1.0；出现 NaN 先降 lr、检查缓存是否损坏（latent NaN）、再检查 caption 是否为空。
- EMA 只在步数足够时有用；短跑（<500 步）收益有限。

## 6. 采样与验证

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

采样 CLI 的完整参数（`--lora`、`--ema`、`--dtype`、`--output-type latent` 等）见 [README](../README.md#6-采样) 与 `moss-lora-sample --help`。

## 7. 相关文档

- 架构、token 数、flow-matching、latent 约定与 LoRA 设计：[RESEARCH.md](RESEARCH.md)
- 32GB 显存规划、OOM 阶梯、compile/SDPA/checkpointing：[5090.md](5090.md)
- 数据目录、layout、caption 规范：[DATASET.md](DATASET.md)
- adapter 转换与 ComfyUI 接线：[COMFYUI.md](COMFYUI.md)
