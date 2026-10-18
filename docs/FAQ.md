# FAQ

短答优先。涉及数字的条目，除注明"外部报告"或"20-step smoke 实测"外均为**估算**；本项目只做过一次 20-step 真实模型 smoke（1.25s/step、12.84GB 峰值），没有长跑或多配置 benchmark。

---

## 安装与硬件

**Q1. RTX 5090 要装什么版本的 PyTorch？**

CUDA 12.8+ 的 wheel。本项目基线是 `torch==2.9.1` + cu128：

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

验证 `torch.cuda.get_device_capability(0) == (12, 0)` 且 `torch.cuda.get_arch_list()` 包含 `sm_120`。默认 PyPI 的 wheel 可能不带 sm_120 kernel。

**Q2. 24GB 显卡能跑吗？**

可能可以，但不在支持范围内。1M 像素、batch 1、checkpointing 的峰值估算 14–16GB；20-step smoke 实测 12.84GB（compile off），24GB 理论上有余量；但 CUDA context、compile、allocator 碎片会吃掉余量。更稳的是 512² 或 768²。请用 `moss-lora-bench` 实测。

**Q3. Windows 能用吗？**

建议 WSL2（Ubuntu 22.04/24.04）。原生 Windows 未验证，多 worker dataloader、`torch.compile`/Triton 等都可能出问题。WSL 下把数据放在 ext4 文件系统，不要放 `/mnt/c`。

WSL2 上另一个高频坑：训练刚启动就报 `CUDA driver error: unknown error`，堆栈在 AdamW `_init_group`，`dmesg` 里有 `dxg get_unused_fd_flags failed ... EMFILE`。这不是驱动或显存问题，而是 WSL 默认 soft `RLIMIT_NOFILE=1024` 被 GPU 初始化/优化器建组耗尽 fd。每次 GPU 运行前执行 `ulimit -n 65536` 再启动；远程/CI 命令要显式带上，写进 `~/.bashrc` 只对交互 shell 生效。

**Q4. 需要 flash-attn / xformers / bitsandbytes 吗？**

不需要。

- attention 用 PyTorch 原生 SDPA；sm_120 上 flash-attn 的预编译轮子经常不可用，源码编译又需要完整 CUDA toolkit。
- optimizer 用 PyTorch fused AdamW；不需要 bitsandbytes 的 8-bit optimizer。
- xformers 不是依赖。

**Q5. 支持多卡吗？**

不支持。本项目只针对单卡 5090；没有 DDP/FSDP 路径，也没有 NVLink 相关优化。

---

## 训练

**Q6. 1M 像素 OOM 怎么办？**

按顺序：确认 `gradient_checkpointing: true`、`batch_size: 1`、训练进程没有加载文本编码器/VAE；设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；关闭 `compile`；降到 768 或 512。完整阶梯见 [5090.md](5090.md#4-oom-阶梯)。

**Q7. 为什么 checkpointing 默认开？**

1M 像素不 checkpoint 的激活约 23GB（估算），加上 11.4GB 固定占用就超过 32GB。checkpointing 把激活降到约 1–2GB，代价是多一次前向（训练计算约 +33%）。512 下 batch 1 可以关掉换速度。

**Q8. 需要训练多少步？**

1M、batch 1 下通常 1–3k 步。更合理的算法是 `epochs × 图片数 / batch_size`：

- 20–100 张风格图：500–1500 步；
- 500+ 张人物/领域图：2–5k 步；
- 1–5k 张稳健概念：3–10k 步。

先看 holdout，不要只看 train loss。

**Q9. LoRA rank 选多少？**

- 16：37.2M 参数，风格/轻量适配；
- 32：74.3M，默认；
- 64：148.6M，复杂概念/文字排版/大数据集。

alpha 取 rank。rank 越大越容易过拟合，显存也更高（rank 64 的 fp32 参数+梯度+AdamW 约 2.38GB）。

**Q10. 学习率多少？**

5e-5 – 2e-4，默认 1e-4。小数据集、rank 64 用下限；大数据集可到 2e-4。warmup 50–200 步，梯度裁剪 1.0。

**Q11. loss 应该降到多少？**

没有通用目标值。loss 是不同 σ 下的 velocity MSE，绝对值随采样分布变化，不能跨配置比较。看趋势（EMA loss）、看 holdout 出图、看文字/构图是否改善。出现 NaN 先降 lr 并检查缓存。

**Q12. 如何 resume？是精确恢复吗？**

```bash
moss-lora-train --config configs/train_5090_1m.yaml --resume runs/moss_1m
```

实现会保存 LoRA、optimizer、scheduler、step 与 RNG 状态；`--resume` 可传 run 目录（自动选最新 `step_*`）或具体 checkpoint 目录。保持同一 config、同一缓存、同一数据顺序。CUDA 非确定性算子可能让结果不是 bit 级一致。

**Q13. EMA 值得开吗？**

步数 ≥1k 时通常值得，代价是额外一份 fp32 LoRA（rank 32 约 0.30GB）和每步更新。短跑（<500 步）收益有限。采样时用 `--ema` 对比 raw 与 EMA 的 holdout。

**Q14. 能不能只训 attention，不加 FFN？**

可以改 `lora.targets`，但不推荐作为默认。FFN 占 DiT 参数约 62%，只训 attention 常见的结果是"风格像了，但内容/纹理不对"。至少保留 `ffn_down` 或 `ffn_up_gate` 做对照。

**Q15. 能不能微调文本编码器？**

不建议。官方 User Guide 明确要求冻结 TE；8.393B 参数的 fp32 训练状态远超 32GB；而且 TE 一变，所有缓存 embedding 失效。需要适配新语言/领域时，离线训练独立 TE adapter 并重建缓存，属于另一条工作流。

**Q16. 可以用 MOSS-Image-SFT 作为 base 吗？**

`model.path` 可以指向 SFT 模型，但本项目的配方是针对 Base 调的。SFT 已经过 post-training，LoRA 的学习率/步数需要重新验证，且可能更快过拟合。

**Q17. 为什么默认不开 FP8 / NVFP4？**

flow matching 回归的是连续 velocity，FP8 尾数位不足会把误差直接带入 MSE，常见表现是细节漂移、文字边缘发虚。官方推理是 bf16 + fp32 CFG/Euler；默认 FP8 会引入与推理不一致的数值路径。除非实现提供可选的 scaled matmul 路径，并用 holdout 做 A/B，否则不要开。

**Q18. `timestep_sampling` 为什么默认 `logit_normal`？**

对齐技术报告 Stage 3 最终阶段的噪声分布：μ=-1.09、s=0.95、cutoff=0.95、1% 均匀混合。分布偏向高噪声端（S 中位数约 0.75），有利于结构/布局学习；1% 均匀保证低噪声端覆盖。`uniform_grid` 仅用于消融。

---

## 数据与缓存

**Q19. 缓存占多少磁盘？**

1M 像素约 2MB latent/图；512² 约 0.5MB；1536² 约 4.5MB。文本 embedding 每图约 1–4MB（取决于长度和变体数）。1k 张 1M 图合计约 3–6GB。

**Q20. 改了分辨率/数据，需要重建缓存吗？**

需要。缓存与图片内容、`target-area`、`latent-dtype`、`max-text-length`、tokenizer revision、负向提示词绑定。任何一项变化都要重建；否则形状/条件不一致。

**Q21. caption 的 long/medium/short 怎么用？**

三个变体都缓存，训练时每个 step 对每张图等概率选一个（1:1:1）。这比只用 long 更能防止模型过拟合到固定句式。

**Q22. caption 可以是空的吗？**

不可以。空 caption 会变成空文本条件，训练出"无条件生成"行为。至少要有一句准确的描述；确实没有可用 caption 时，用官方 caption prompt 重新生成。

**Q23. 缓存报 `ValueError: Caption is empty or unsupported: None` 怎么办？**

说明某张图没有可用 caption：sidecar 缺失、`.txt` 为空/纯空白，或 `.json` 用了未识别的字段（可用 `caption`/`text`/`prompt`/`image_description`，或官方 `layout`）。默认 `--missing-caption skip` 会打印 `warning: skipping <图片路径>: ...` 并跳过，最后汇总 `N images, M skipped (no/empty caption)`；`--missing-caption error` 会在第一张坏图处带路径失败。先检查 sidecar 是否与图片同名（`0001.jpg` ↔ `0001.txt`/`0001.json`）且非空；修好后重跑。不要长期依赖 skip：空 caption 会退化出无条件生成行为；全部图片都被跳过时缓存会直接报错。

**Q24. layout 的 bbox 总是错位？**

检查三件事：格式必须是 `yxyx`；坐标必须是 `abs_centralized`（减去 W/2、H/2）；画布尺寸必须与缓存时的 bucket 一致。缓存裁剪会改变画布，文字密集数据建议固定正方形单 bucket。

---

## 采样与质量

**Q25. 出图质量差、过饱和、文字糊？**

先确认采样参数是官方默认：≥50 步、CFG 3.0、shift 5.0、Euler、负向分支 skip layer 22、官方负向提示词。低于 50 步或 CFG 过高是最常见原因。其次检查训练分辨率（文字密集建议 1.5M+）和 caption 是否逐字描述了文字。

**Q26. LoRA 效果太强/太弱？**

太强：降 lr、减步数、降 rank；部署时降低 LoRA scale。太弱：提高 lr（不超过 2e-4）、增加步数、确认目标层包含 FFN、确认缓存与训练条件一致。alpha=rank 是基线。

**Q27. CFG 产生伪影怎么办？**

base 模型对 CFG 敏感。用 3.0 起步；短 prompt 可升到 4–6，长 prompt 保持 3.0；开启 skip layer 22；换 seed 排除偶发样本。不要用蒸馏模型常见的 CFG 1.5–2.0 设置。

**Q28. merge 还是运行时加载 LoRA？**

迭代/对比用运行时 LoRA（`moss-lora-sample --lora ...`）；部署用 `moss-lora-merge` 合并成只含 transformer 的目录，官方 pipeline 可将其作为 transformer 组件加载。用本仓库 CLI 时继续传 base 模型 `--model`，并加 `--transformer merged/`（`--model` 仍提供文本编码器、tokenizer、VAE）。合并不可逆，保留原始 adapter。

**Q29. 怎么判断过拟合？**

固定 holdout、固定 seed，每 250–500 步出图。train loss 继续降但 holdout 的文字准确率/构图/风格变差，就是过拟合。优先降步数/rank/lr，增加 caption 多样性，必要时加 LoRA dropout 0.05。

---

## 其他

**Q30. 许可证是什么？**

本仓库代码 Apache-2.0。MOSS-Image 模型与官方推理代码 Apache-2.0。Qwen3.5 文本编码器、FLUX.2 VAE 等组件遵循各自上游许可，商用前请分别确认。

**Q31. 有官方训练代码吗？**

没有。OpenMOSS 发布了模型、推理 pipeline、技术报告和 User Guide，但没有发布训练代码。本项目的配方是从这些材料推导的，LoRA 超参是工程选择。第三方社区训练器 `mrfakename/moss-image-trainer` 可作对照，但不是官方实现。

**Q32. 为什么文档里的显存/速度都是"估算"？**

因为目前只有一次 20-step 真实模型 smoke：RTX 5090 32GB、16 张 1M 图、rank 32、batch 1、checkpointing on、compile off；steady **1.25 s/step**（首次 2.59s），峰值 **12.84GB**，loss mean 0.423，训练墙钟 46.6s。这是冒烟数据，不是长跑/多配置基准；采样也只跑了 8 步。其余估算用参数量、token 数、FLOPs 和 5090 规格算出，用于容量规划；实际值受驱动、kernel、compile、文本长度和 allocator 影响。请用 `moss-lora-bench` 在自己的机器上测，并填入 [5090.md](5090.md) 的表格。

**Q33. 训练出的 LoRA 可以直接商用吗？**

取决于三件事：基础模型/组件的许可证、训练数据的版权与肖像权、以及你所在地区的合规要求。代码许可证不覆盖模型权重和数据的权利。
