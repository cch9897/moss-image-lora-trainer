# MOSS-Image LoRA 训练研究笔记

本文记录本项目训练配方的推导过程：从官方模型配置、`pipeline_moss_image.py`、技术报告和 User Guide 出发，解释每一个工程选择，并给出可复核的显存/吞吐模型。

显存/吞吐以解析估算为主，另有 2026-10-18 的 **20-step smoke 实测**（§10.3，1.25s/step、12.84GB）。估算用于容量规划，不是性能声明；20 步短跑也不能外推长训练。

---

## 0. 结论摘要

1. MOSS-Image-Base 的 DiT 是 5.082B 参数的单流 Transformer，bf16 约 10.16GB；文本编码器 Qwen3.5-9B 为 8.393B 参数，bf16 约 16.79GB。两者不可能在 32GB 卡上同时常驻训练。
2. 因此训练必须把文本编码与 VAE 编码离线化：训练循环只保留 DiT + LoRA。这是本项目最大的显存与速度收益。
3. 训练目标是 rectified flow / flow matching：`v = ε - x0`，`x_t = (1-σ)x0 + σε`，模型时间步 `t = 1000σ`，fp32 MSE，无 sigma 加权。
4. σ 默认**不是**均匀网格，而是报告 Stage 3 最终阶段的**截断 logit-normal**（μ=-1.09，s=0.95，cutoff=0.95，1% 均匀混合）在 1000 步离散网格上的采样。
5. 1M 像素、batch 1、rank 32、checkpointing 的峰值显存估算为 **14–16GB**；20-step smoke 实测 **12.84GB**（compile off）。不开 checkpointing 约 33–36GB，32GB 下大概率 OOM。
6. 5090 的 bf16 稠密峰值 209.5 TFLOPS，低于 A100 80GB 的 312 TFLOPS；1792GB/s 的 GDDR7 带宽也低于 A100 80GB SXM 的 2039GB/s。它适合本项目的原因是 32GB 显存、FP4 和消费级成本，而不是 bf16 绝对算力。

---

## 1. 事实来源与版本

| 事实 | 来源 |
| --- | --- |
| DiT 超参（42 层、3072、24 heads、FFN 8192、128 latent channels、text_dim 4096、freq_dim 256、rope_base 10000） | 模型仓库 `transformer/config.json` 与 `pipeline_moss_image.py` |
| DiT 参数量 5,082,308,608；bf16 权重 10,164,617,280 字节 | [模型仓库 safetensors index](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) |
| 文本编码器 8,392,695,024 参数；16,785,390,048 字节 | 模型仓库 `text_encoder/model.safetensors.index.json` |
| VAE `AutoencoderKLFlux2`，32 latent channels，8× 下采样，patch 2×2 | 模型仓库 `vae/config.json` |
| Flow-matching 损失、无 sigma 加权、Stage 3 噪声表 | [技术报告](https://github.com/OpenMOSS/MOSS-Image/blob/main/paper/MOSS_Image_Technical_Report.pdf) 附录 A.2，Eq. (3)、Table 6 |
| AdamW β=(0.9, 0.95)、ε=1e-8、wd=0.01、clip=1.0 | 技术报告 Table 7 |
| 推理默认（50 步、CFG 3.0、shift 5.0、Euler、skip layer 22、负向提示词） | 模型仓库 `scheduler/scheduler_config.json`、pipeline、User Guide |
| TE 冻结、caption prompt、推荐 VLM | [User Guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) |
| 5090 规格（21760 CUDA、680 tensor cores、32GB GDDR7、1792GB/s、bf16 209.5 TFLOPS） | [NVIDIA RTX Blackwell 架构白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) 附录 A |

**明确缺失**：OpenMOSS 没有发布官方训练代码，也没有发布官方 LoRA 超参。除报告中的预训练设置外，LoRA 的 rank、目标层、lr、步数都是本项目的工程选择。报告的训练规模是 512 卡、百万级 step，不能直接外推到单卡 LoRA。

---

## 2. 架构拆解

### 2.1 DiT

| 组件 | 配置 | 备注 |
| --- | --- | --- |
| 层数 | 42 | 不含 text adapter 的 2 层 |
| hidden | 3072 | |
| attention | 24 heads × 128 | 单流 joint self-attention |
| FFN | 8192，SiTU gate | `4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)`，即 β_g=4、β_u=25 |
| modulation | 共享 6-way | 每个 block 共用一组 shift/scale/gate |
| 位置编码 | 4D RoPE，base 10000 | 轴为 (ref, t, w, h)；ref=0 时前 32 维不旋转 |
| latent channels | 128 | patchify 后的通道数 |
| text_dim | 4096 | Qwen3.5 hidden |
| freq_dim | 256 | timestep sinusoid；fp64 计算、fp32 送入 MLP |

`forward` 的输入 `hidden_states` 形状是 `[B, 128, H/16, W/16]`，即已经 patchify 的 latent，不是 VAE 原始 32 通道。1024×1024 对应 `[B, 128, 64, 64]`。

### 2.2 参数都花在哪

每个 block：

- q/k/v/o：`4 × 3072² = 37,748,736`
- FFN：`3 × 3072 × 8192 = 75,497,472`
- 合计 `113,246,208` / block

42 个 block：`4,756,340,736`（约 4.76B）。

其余约 0.326B：

| 部分 | 参数量（约） |
| --- | --- |
| text adapter（4096→3072 + 2 个 3072/8192 block） | 239.5M |
| video embedder（128→3072） | 0.39M |
| time embedding（256→3072→3072） | 10.2M |
| shared modulation（3072→18432） | 56.6M |
| final norm + final linear（3072→128） | 0.40M |
| head modulation（3072→6144） | 18.9M |
| 各类 RMSNorm | 0.28M |

总计约 5.082B，与 safetensors 元数据一致。FFN 占约 62%，attention 占约 31%；这也是 LoRA 必须覆盖 FFN 的原因。

### 2.3 冻结组件

| 组件 | 参数量 | bf16 体积 | 是否训练 |
| --- | --- | --- | --- |
| DiT | 5.082B | 10.16GB | 只训练 LoRA |
| Qwen3.5-9B | 8.393B | 16.79GB | 冻结；仅缓存阶段使用 |
| VAE | ~62M（文件 249.5MB，含 fp32） | 0.25GB | 冻结；仅缓存/采样使用 |

Qwen3.5 的配置是 32 层、hidden 4096、16 个 Q heads / 4 个 KV heads、head_dim 256；每 4 层一个 full attention，其余为 linear attention（GDN）。这解释了为什么它在 4096 token 下的显存仍然可控，但 16.79GB 的权重对 32GB 训练仍是硬约束。

---

## 3. Token 与序列长度

VAE 下采样 8×，再 2×2 patch，等效图像 stride 为 16：

```
N_img = (H / 16) × (W / 16) = H×W / 256
```

| 分辨率 | 像素 | VAE latent | patch 网格 | image tokens |
| --- | --- | --- | --- | --- |
| 512×512 | 0.262M | 64×64 | 32×32 | 1,024 |
| 768×768 | 0.590M | 96×96 | 48×48 | 2,304 |
| 1024×1024 | 1.049M | 128×128 | 64×64 | 4,096 |
| 1216×1216 | 1.479M | 152×152 | 76×76 | 5,776 |
| 1280×1280 | 1.638M | 160×160 | 80×80 | 6,400 |
| 1536×1536 | 2.359M | 192×192 | 96×96 | 9,216 |

文本 token 数可变：缓存 CLI 的截断上限默认 1024，官方上限 4096；示例配置用 512 控制成本。序列总长 `S = N_img + N_text`。1024² + 512 token 时 `S = 4608`；1024² + 4096 token 时 `S = 8192`。

> 文本 token 参与 joint attention，但不参与 loss。report Eq. (3) 只对 image token 求和。

**长 caption 的代价不是线性的**：投影/FFN 随 S 线性增长，attention 随 S² 增长。1024² 下把文本从 512 提到 4096，前向 FLOPs 从约 51 TFLOPs 增到约 112 TFLOPs（估算，见 §10）。这是示例/推荐配置用 512 控制成本的原因（CLI 默认 1024）；layout/长文本数据集再单独调高。

---

## 4. Flow matching

### 4.1 前向加噪与目标

记 `x0` 为归一化后的 latent，`ε ~ N(0, I)`，`σ ∈ (0, 1]`：

```
x_σ = (1 - σ) · x0 + σ · ε
v   = ε - x0
t   = 1000 · σ
```

模型预测 velocity `vθ(x_σ, t, c)`，损失是 image token 上的 fp32 MSE：

```
L = mean_{image tokens, channels} ( vθ - (ε - x0) )²
```

关键点：

- **没有 sigma 加权，也没有 SNR 加权**（report Eq. 3 明确说明）。
- loss 在 autocast 外以 fp32 计算；bf16 只用于前向 GEMM。
- 文本 token 不参与 loss。
- 时间步是 `1000σ`，不是 `σ`。原因是 DiT 的 timestep sinusoid 与 scheduler 都以 0–1000 为刻度；直接把 `σ∈(0,1]` 传进去会落在频率范围的最低端，和预训练时的条件分布不匹配。
- 目标是 `v = ε - x0`，即插值路径对 σ 的导数：`dx_σ/dσ = ε - x0`。它在整条直线路径上是常量，因此每个 σ 都有定义良好、量级一致的目标；相比直接预测 ε 或 x0，不需要额外的 SNR 加权来平衡不同噪声档位。

### 4.2 为什么用离散网格

训练网格是 `σ_j = j / 1000`，`j ∈ {1, ..., 1000}`。模型看到的时间步就是整数 `j`。这与官方 scheduler 的 `training_shift=1.0` 一致：训练阶段没有对 σ 做 shift；shift 只在采样阶段作用于推理网格。

### 4.3 σ 采样：截断 logit-normal（默认）

报告 Stage 3 的最终阶段（274,001–310,000 步）使用：

```
μ = -1.09, s = 0.95, cutoff a = 0.95, uniform mix ρ = 0.01
Z ~ N(μ, s²)
S = 1 - sigmoid(Z)
拒绝采样直到 S ≤ a
j = clamp(ceil(S · 1000 / a), 1, 1000)
σ = j / 1000
```

以 1% 概率走均匀分支：

```
j ~ Uniform{1, ..., 1000}
σ = j / 1000
```

报告 Eq. (4) 给出的离散概率为：

```
p_j = (1-ρ) · [F(a·j/T) - F(a·(j-1)/T)] / F(a) + ρ/T
```

其中 `F` 是 `S` 的 CDF，`T=1000`。实现上把裁剪后的分布重标到完整网格 `{0.001, ..., 1.000}`，所以 σ 仍可到 1.0。

**分布形状的直觉**：`sigmoid(-1.09) ≈ 0.251`，因此 `S = 1 - sigmoid(Z)` 的中位数约 **0.749**。分布明显偏向高噪声端（结构/布局形成阶段），再叠加 1% 均匀采样保证低噪声端不被完全忽略。这也是默认不用均匀网格的原因：σ 上的均匀采样会把大量步数花在低噪声端，而结构/布局主要在高噪声端形成。这是预训练最后阶段的配方，本项目把它作为 LoRA 默认值；它不是唯一正确选择，`timestep_sampling: uniform_grid` 可用于消融。

> 历史注意：不要把"uniform integer grid t∈[1,1000]"当作默认。报告前两个 Stage 3 区间使用不同的 `(μ, s)` 和 shift，且最初区间有 1% 的 full-noise override；LoRA 采用的是**最终区间**的截断 logit-normal。

### 4.4 采样（推理）

官方 scheduler 的推理过程：

1. 在 50 步上构造 shift=5.0 的网格：

```
lower = shift / (T + shift - 1)
values = linspace(1, lower, steps)
σ = shift · values / (1 + (shift - 1) · values)
```

2. 把每个 σ 对齐到训练网格 `{j/1000}` 上最近的点（`searchsorted` + 取最近），去重。
3. Euler 更新：`x ← x + (σ_next - σ) · vθ`。
4. CFG 在 fp32 中合并：`v = v_uncond + w · (v_cond - v_uncond)`；负向分支在 skip layer 22 处整块跳过（仍做两次前向）。
5. 最终 latent 为 fp32；VAE decode 前 denormalize → unpatchify。

这与 `pipeline_moss_image.py` 的 `MOSSImageScheduler` / `MOSSImagePipeline` 完全一致。

---

## 5. Latent 预处理：patchify 与 BatchNorm

### 5.1 为什么顺序很重要

VAE 输出 `[B, 32, H/8, W/8]`。官方 pipeline 在**patchify 之后**用 VAE 的 BatchNorm running stats 归一化 **128 通道**，而不是先对 32 通道归一化。`AutoencoderKLFlux2` 的 BN 定义为 `BatchNorm2d(prod(patch_size) * latent_channels) = BatchNorm2d(128)`。

### 5.2 编码（缓存阶段）

```
p = lat.view(B, 32, H/16, 2, W/16, 2)
       .permute(0, 1, 3, 5, 2, 4)
       .reshape(B, 128, H/16, W/16)

z = (p - bn.running_mean) / sqrt(bn.running_var + eps)
```

`eps = batch_norm_eps = 1e-4`。缓存默认存 fp32（`--latent-dtype bf16` 可减半，但会损失一点精度）；训练时再按需转 bf16 计算。对 1024²，`z` 的形状是 `[128, 64, 64]`，即每图 2,097,152 字节（fp32）。

VAE encode 使用 `latent_dist.mode()`（确定性），不是采样。这样缓存可复现，且与推理时的均值路径一致。

### 5.3 解码（采样阶段）

```
p = z * sqrt(bn.running_var + eps) + bn.running_mean

u = p.reshape(B, 32, 2, 2, H/16, W/16)
     .permute(0, 1, 4, 2, 5, 3)
     .reshape(B, 32, H/8, W/8)

x = vae.decode(u)
```

然后 `clamp(-1, 1)` 并转 PIL。顺序不能反：先 denormalize，再 unpatchify，最后 VAE decode。

### 5.4 缓存体积

| 分辨率 | latent 形状 | fp32/图 | 1,000 图 |
| --- | --- | --- | --- |
| 512² | 128×32×32 | 0.50MB | 0.5GB |
| 1024² | 128×64×64 | 2.00MB | 2.0GB |
| 1536² | 128×96×96 | 4.50MB | 4.5GB |

文本 embedding 是 `[L, 4096]` bf16；512 token 约 4.19MB，128 token 约 1.05MB。加上三个 caption 变体与 layout JSON，实际每图磁盘占用约 **2–6MB**。

---

## 6. 文本编码

官方 pipeline 的 `encode_prompt`：

```python
formatted = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
inputs = tokenizer(formatted, return_tensors="pt",
                   truncation=True, max_length=max_sequence_length)
output = text_encoder(**inputs, use_cache=False, return_dict=True)
emb = output.last_hidden_state[0]      # [L, 4096], bf16
```

要点：

- **逐条编码、不 padding**。pipeline 注释明确说 padding 会影响 Qwen GDN 的输出；缓存阶段逐 caption 前向，保持真实长度。
- `add_generation_prompt=True` 与 `enable_thinking=True` 必须同时保留；换 tokenizer revision 会改变 embedding。
- 取 `last_hidden_state`，不是 pooler output。
- 训练 collate 可以把不同长度 padding 到 batch 最大值，但 DiT 内部按 mask 逐样本切片，padding token 不参与 attention，也不参与 loss。缓存阶段仍然逐条编码，避免 padding 污染。
- 上限：官方 4096；缓存 CLI 默认 1024，示例配置用 512。layout JSON 建议 1024–2048。

**为什么冻结 TE**：

1. 官方 User Guide 明确要求 "keep the text encoder frozen"。
2. 8.393B 参数，fp32 训练需要 ~33.6GB 权重 + 梯度 + Adam 状态，单卡 32GB 不可能。
3. 一旦 TE 变化，所有缓存 embedding 失效；每步重新编码 9B 模型会吃掉全部收益。
4. 文本条件漂移会破坏 base 模型的 prompt 泛化，尤其是文字渲染与 layout。

如果确实需要适配新语言/新领域词汇，正确做法是离线训练一个独立的 TE adapter，然后**重建全部缓存**；这不在本项目默认路径内。

---

## 7. LoRA 设计

### 7.1 目标层

每个 DiT block：

| 目标 | 形状 | LoRA 参数量 `r(in+out)` |
| --- | --- | --- |
| q | 3072→3072 | `r × 6144` |
| k | 3072→3072 | `r × 6144` |
| v | 3072→3072 | `r × 6144` |
| o | 3072→3072 | `r × 6144` |
| ffn_up_gate | 3072→16384（fused up+gate） | `r × 19456` |
| ffn_down | 8192→3072 | `r × 11264` |

每 block 合计 `r × 55296`；42 blocks：

| rank | 每 block | 总计 | fp32 参数 | 参数+梯度+AdamW |
| --- | --- | --- | --- | --- |
| 16 | 884,736 | 37,158,912 | 148.6MB | ~0.59GB |
| 32 | 1,769,472 | 74,317,824 | 297.3MB | ~1.19GB |
| 64 | 3,538,944 | 148,635,648 | 594.5MB | ~2.38GB |

> 如果把 `ffn_up_gate` 误按 8192 输出维度估算，会得到 rank 32 = 63.3M；官方 fused 层是 16384 输出，实际是 74.3M。训练启动日志会打印真实可训练参数量。

### 7.2 为什么选这些层

- **q/k/v/o**：控制内容选择、空间对应和残差写入。风格/构图适配主要靠这里。
- **FFN**：占 DiT 参数 62%，承载领域知识与纹理先验。只训 attention 的 LoRA 往往"风格像了但内容不对"。
- **不训 norm/modulation/time embedding**：参数量小、对全局尺度敏感，LoRA 的收益低、破坏性大。
- **text adapter 默认不加**：它负责把 4096 维文本投射到 3072；在缓存固定的前提下，改它等于改变文本条件的解释方式，容易导致过拟合和 layout 漂移。`lora.text_projection` 提供开关，但默认关闭。
- **final linear 默认不加**：输出层直接决定 velocity 尺度，小数据上风险高。

### 7.3 初始化与缩放

标准 LoRA（B 零初始化）：`W' = W + (alpha / r) · B A`，A 随机初始化、B 为零。因此**第 0 步的 LoRA 输出为零**（在实现采用 B=0 初始化时），采样结果应与 base 一致（在数值误差内）。这提供了一个很好的自检：训练前用 `--lora` 指向一个未训练的 adapter，出图应与 base 相同。

alpha=rank 时缩放为 1；这是最不容易出错的默认。alpha 大于 rank 相当于放大 LoRA 更新，短跑可能更快见效，但也更容易过拟合。

### 7.4 EMA

EMA 保存一份 LoRA 参数的指数滑动平均。技术报告的预训练使用 EMA 0.9999；LoRA 训练步数少得多，0.999 更合适。EMA 的代价是额外一份 fp32 LoRA（rank 32 约 0.30GB）和每步一次原地更新。步数 <500 时收益有限。

---

## 8. 精度策略

| 位置 | 精度 | 理由 |
| --- | --- | --- |
| DiT 基座权重/激活 | bf16 | 与官方推理一致；fp16 range 不足 |
| LoRA 参数 | fp32 | 小参数量，fp32 避免更新被 bf16 舍入吃掉 |
| 梯度 | fp32 | 与 LoRA 参数一致 |
| AdamW 状态 | fp32 | 标准 |
| velocity loss | fp32 | MSE 对尾部误差敏感；bf16 loss 会掩盖尖峰 |
| timestep sinusoid | fp64 计算 → fp32 MLP | 与官方 `forward` 一致 |
| latent 缓存 | fp32 | 保留 VAE 输出精度；训练时再 cast |
| VAE decode（采样） | fp32 | 官方默认；bf16 decode 会有可见色带 |

### 为什么默认不开 FP8/NVFP4

- flow matching 的回归目标是连续 velocity；FP8 的尾数位不足，误差会直接进入 MSE，表现为细节漂移、文字边缘发虚。
- 官方推理是 bf16 + fp32 CFG/Euler；训练用 FP8 会引入与推理不一致的数值路径。
- sm_120 的 FP8/FP4 tensor core 需要额外的 scaled matmul / Transformer Engine 路径，当前实现没有默认启用。
- 如果未来实现支持，应作为**可选实验**，并用固定 holdout 与 bf16 基线做 A/B，而不是默认开启。

---

## 9. 显存模型

### 9.1 固定占用（与分辨率无关）

| 项 | 估算 |
| --- | --- |
| DiT bf16 权重 | 10.16GB |
| LoRA fp32 参数（r32） | 0.30GB |
| LoRA 梯度（r32） | 0.30GB |
| AdamW m/v（r32） | 0.59GB |
| CUDA context / cuBLAS/cuDNN workspace / allocator 余量 | 1.0–2.0GB |
| **小计** | **~12.4–13.4GB** |

README 的表格用 ~11.4GB 表示纯张量部分（10.16+1.19），不含 context；两处口径不同，规划时以 12–13GB 为固定成本更安全。

### 9.2 激活（1M 像素，batch 1）

取 `S = 4096 + 256 = 4352`，bf16。每个 block 在前向中会被 autograd 保存的主要张量（估算）：

| 张量 | 大小 |
| --- | --- |
| residual 输入 x | 26.7MB |
| norm1 输出 / q,k,v | 26.7 + 80.1MB |
| attention 输出 / o 投影 | 26.7 + 26.7MB |
| 第二个 residual / norm2 | 26.7 + 26.7MB |
| ffn_up_gate 输出（16384） | 142.6MB |
| SiTU 输出（8192） | 71.3MB |
| ffn_down 输入/输出 | 71.3 + 26.7MB |
| **每 block 约** | **~0.55GB** |

42 个 block：**~23GB**。加上固定占用，1M 不 checkpoint 的峰值约 **35–36GB**，32GB 下 OOM。

开启 checkpointing 后只保存每个 block 的输入 `x`：`42 × 26.7MB ≈ 1.12GB`；反向时逐块重算，瞬时峰值约一个 block 的 0.55GB。加上 embedding、RoPE、临时 buffer，激活约 **1.5–2.5GB**。总峰值 **14–16GB**，与社区在 A100 上报告的 15.7GB 同一量级（外部数据，仅作 sanity check）。

### 9.3 为什么 TE/VAE 必须离线

如果训练时同时加载：

```
DiT 10.16 + TE 16.79 + VAE 0.25 + LoRA/opt 1.19 + 激活 1.5
≈ 29.9GB（张量本身）
```

再加 CUDA context 和 allocator 碎片，32GB 直接见底；而且每步还要为 8.4B 模型做前向。缓存后，训练循环的固定成本回到 ~11.4GB，才有多余空间开 batch、compile 或 EMA。

采样阶段同样不能三者共存。`moss-lora-sample` 的顺序是：TE 编码 prompt → 卸载到 CPU（或释放）→ DiT + VAE 上 GPU。峰值从 >28GB 降到约 12–14GB。

---

## 10. 吞吐模型

### 10.1 每 token 的 FLOPs

每个 DiT block、每个 token（MACs）：

```
attention 投影 + FFN:
  4·d² + 3·d·f = 4×3072² + 3×3072×8192
              = 37,748,736 + 75,497,472
              = 113,246,208 MACs

joint attention（双向，QK^T + AV）:
  2·S·d MACs
```

乘以 2 得 FLOPs，再乘以 token 数和 42 层。

### 10.2 关键配置的估算

| 配置 | S | 前向 FLOPs | 训练 FLOPs（无 ckpt / 有 ckpt） |
| --- | --- | --- | --- |
| 512², text 256 | 1,280 | 13.0 TFLOPs | 39.1 / 52.1 TFLOPs |
| 1024², text 256 | 4,352 | 51.2 TFLOPs | 153.5 / 204.7 TFLOPs |
| 1024², text 512 | 4,608 | 54.8 TFLOPs | 164.4 / 219.2 TFLOPs |
| 1024², text 4096 | 8,192 | 112.5 TFLOPs | 337.6 / 450.1 TFLOPs |
| 1536², text 256 | 9,472 | 136.4 TFLOPs | 409.1 / 545.5 TFLOPs |

训练 FLOPs 的口径：前向 + 反向 ≈ 3× 前向；checkpointing 多一次前向 ≈ 4×。未计 optimizer、数据搬运和 kernel 低效。

### 10.3 时间估算

5090 的 bf16 tensor 稠密峰值（FP32 accumulate）是 **209.5 TFLOPS**。DiT 训练的实际 MFU 取决于 kernel、attention backend、compile 和形状；这里取 25% / 35% 两个档位做区间：

| 配置 | 有效算力 52.4 TFLOPS（25%） | 73.3 TFLOPS（35%） |
| --- | --- | --- |
| 1024², batch 1, ckpt | ~3.9s/step | ~2.8s/step |
| 1024², batch 1, 无 ckpt | ~2.9s/step | ~2.1s/step |
| 512², batch 1, ckpt | ~1.0s/step | ~0.7s/step |
| 512², batch 4, ckpt | ~4.0s/step | ~2.8s/step |
| 1536², batch 1, ckpt | ~10.4s/step | ~7.4s/step |

**这些是解析估算，不是实测。** 2000 步 1M 训练对应约 1.6–2.2 小时（按 2.8–3.9s/step），不含缓存、验证采样和保存。

**2026-10-18 的 20-step smoke 实测**（仅作冒烟参考，不是长跑 benchmark）：本地 WSL2 机器，RTX 5090 32GB（32607 MiB），驱动 610.57.01，`torch 2.14.1+cu130`，47GiB RAM；16 张 1M 图，`--max-text-length 2048`，fp32 latents，cache 55MB/33.7s；rank 32，bf16，batch 1，gradient checkpointing on，compile off。结果：**1.25 s/step** steady（前 2 步含 warmup，首次 2.59s），峰值 **12.84GB**，loss mean 0.423，训练墙钟 46.6s。采样 1024²、8 步、CFG 3、shift 5、skip 22，PNG 656,815 bytes，44s（含模型加载）。

这次实测低于上面的解析区间（2.8–3.9s/step），说明估算偏保守；但只有 1 个 seed、16 张图、20 步、8 步采样，**不要**用它外推 2000 步或画质收敛。WSL2 的 `EMFILE` 坑见 [5090.md](5090.md#7-windows--wsl2-补充)。

外部参考：第三方社区训练器 `mrfakename/moss-image-trainer` 报告 A100 80GB 上 1024px、batch 1、rank 16、checkpointing 约 **1.56s/step**，峰值 **15.7GB**。A100 的 bf16 稠密峰值是 312 TFLOPS，且实现/后端不同，**不能**直接拿来推断 5090 的速度；它只用于校验显存模型的数量级。

### 10.4 attention、GEMM 与 kernel

- 1024² + 256 text 时，attention 约占前向 FLOPs 的 19%（9.8 / 51.2 TFLOPs）；其余是 GEMM。因此编译器和 GEMM 效率比"注意力优化"更重要。
- SDPA 在 sm_120 上可走 flash/cuDNN 融合后端，避免物化 `S×S` 矩阵；`math` 后端会显著变慢并增加显存。
- checkpointing 让计算 +33%，但把激活从 ~23GB 降到 ~1–2GB，是 1M 训练的必需品。
- `torch.compile(dynamic=True)` 主要收益在 pointwise 融合和减少 launch；文本长度变化大时必须用 dynamic，否则每个新长度都重编译（或命中 guard 失败）。
- 数据侧：latent 已缓存，dataloader 只是读文件 + collate；`num_workers=4` 足够。瓶颈通常在 GPU，不在 CPU。
- **逐样本 forward**：官方 `MOSSImageTransformer.forward` 对 batch 内每个样本单独切片 text、单独跑 42 层（`for index in range(batch)`），因此 batch>1 不会自动获得常规 batched GEMM 的吞吐收益；它主要减少 optimizer step 数、改善梯度估计。若训练实现改写成 batched 路径，收益需要实测，不能按线性外推。本文的时间估算按逐样本执行计算。

---

## 11. 质量保障清单

1. **TE 冻结**（官方要求）。
2. **VAE 冻结**，encode 用 `mode()`，decode 用 fp32。
3. **chat template 完全一致**：`add_generation_prompt=True`、`enable_thinking=True`、无 padding、`last_hidden_state`。
4. **latent 归一化顺序一致**：patchify → 128 通道 BN；解码 denormalize → unpatchify。
5. **loss 用 fp32**，不做 sigma 加权。
6. **默认 bf16**，不启用 FP8/FP4。
7. **验证采样对齐官方参数**：50 步、CFG 3.0、shift 5.0、skip layer 22、官方负向提示词；日常监控可降到 30 步省时间，画质对比时用 50 步。
8. **caption 质量**：文字逐字、排版、构图、风格；不要只写标签堆砌。见 [DATASET.md](DATASET.md)。
9. **holdout 固定**：4–8 个样本、固定 seed，训练前先出基线。
10. **EMA 可选**，但比较 raw/EMA 后再决定发布哪个。

---

## 12. 未知与验证清单

| 未知 | 现状 | 如何验证 |
| --- | --- | --- |
| 官方 LoRA 超参 | 未发布 | 本项目的 rank/lr/步数是工程选择；用自己的 holdout 做网格 |
| 最优 σ 分布 for LoRA | 未知 | 默认沿用报告最终阶段；可对比 `uniform_grid` 与不同 μ/s |
| 训练期文本长度上限 | 官方 4096；缓存 CLI 默认 1024、示例用 512 | layout/长文本数据调高，观察显存与文字准确率 |
| LoRA 对 layout 控制的影响 | 未验证 | 用固定 layout JSON + 固定 seed 做前后对比 |
| EMA 对 LoRA 的收益 | 未知 | 同一步数比较 raw/EMA 的 holdout |
| 5090 实际 step time / 峰值 | 仅 20-step smoke：1.25s/step、12.84GB（compile off，16 图，单 seed） | 用 `moss-lora-bench` 做长跑/多配置，填入 [5090.md](5090.md) |
| 长训练（>5k 步）是否过拟合 | 未知 | 每 500 步 holdout；看文字/风格漂移 |
| 与官方推理的数值 parity | 官方明确不要求 bit 级一致 | 用相同 latent/seed 比较输出；允许 kernel 差异 |

---

## 13. 参考链接

- 模型卡：https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
- 官方仓库与 User Guide：https://github.com/OpenMOSS/MOSS-Image
- Pipeline 源码：https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/resolve/main/pipeline_moss_image.py
- 技术报告：https://github.com/OpenMOSS/MOSS-Image/blob/main/paper/MOSS_Image_Technical_Report.pdf
- NVIDIA Blackwell 白皮书：https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- PyTorch cu128 wheels：https://download.pytorch.org/whl/cu128
- 第三方训练器（外部对照）：https://huggingface.co/spaces/mrfakename/moss-image-trainer
