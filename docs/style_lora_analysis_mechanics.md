# MOSS-Image LoRA 条件与缓存机制分析

> 分析对象：`/home/user/projects/MOSS/MOSS-Lora-Trainer`（只读分析；本文件是唯一新增文件）。
> 结论先行：**缓存里没有“artist 权重”这个可调字段**。缓存保存的是冻结 Qwen3.5 文本编码器对每条 caption 的逐 token `last_hidden_state` 和长度；DiT 通过 **joint self-attention**（不是 cross-attention）读取整段文本 token。所谓“缓存时加强 artist”，在当前代码里只有两条路：改 caption 文本，或直接改缓存张量；前者效果弱且会改变分布，后者的乘性尺度会被 DiT 的逐 token RMSNorm 大部分抵消，剩下的只是残差幅度/线性 bias 引起的方向扰动，不构成注意力加权，并且无法在导出的 LoRA / ComfyUI 推理中复现。真正有效的干预点是 **caption 构造、artist dropout/CFG、训练容量**，不是缓存。

证据标记：

- `[代码]`：仓库源码直接可查，附 `file:line`。
- `[实测]`：本机 `/mnt/data/MOSS-Image-Base` 的 tokenizer / 预训练 `text_embedder`、`layers.0` 权重上做的运行时测量。
- `[推断]`：由代码/日志推导，但没有直接观测。
- `[未验证]`：本工作区无法访问对应文件（例如 `/home/user/cache/clean30_v4_1m` 不存在）。

---

## 0. 关键发现一览

| # | 结论 | 证据 |
| --- | --- | --- |
| 1 | 缓存的是 **逐 token hidden states** `[L, 4096]`，不是 pooled embedding；没有 per-token weight，没有 mask | `moss_lora/cache.py:629-634`；`moss_lora/text.py:29-38` |
| 2 | 每个 item 存 `text_0..text_{K-1}` + `text_lengths`；`K` 是该 item 实际存在的 variant 数（1–5，去重后可能少于 5） | `moss_lora/cache.py:307-332, 629-634` |
| 3 | DiT **没有 cross-attention**：文本先经 `text_embedder`（4096→3072 线性 + 2 个 text block），再与 image token `torch.cat`，一起过 42 个 joint block | `moss_lora/model/moss_image.py:145-155, 290-291` |
| 4 | `encoder_attention_mask` 不是传统 attention mask：每个样本先按 mask 取真实 token（`context = ...[mask[index].bool()]`），padding 根本不进入注意力；因此 mask 只负责“去掉 padding”，不能恢复截断，也不能做加权 | `moss_lora/model/moss_image.py:283-285`；`moss_lora/dataset.py:177-195` |
| 5 | loss 只对 image token 计算；text token 是 context，不参与 loss | `moss_lora/model/moss_image.py:299`；`docs/RESEARCH.md:109` |
| 6 | 默认 variant 权重不是均匀：`original 0.16`，`long/medium/short 各 0.64/3`，`layout 0.20`；缺失 kind 按存活项重新归一化 | `moss_lora/cache.py:61-67, 78-101` |
| 7 | 观测到的 variant 直方图（original 17.7%、layout 13.2%）与“约 66% item 有 layout、34% 没有 layout”的混合比例吻合，而不符合“所有 item 都有 5 个 variant” | `docs/bs8_5ep_status.md:84`；本文 §3 `[推断]` |
| 8 | `max_text_length` 代码默认 **1024**，不是 512；512 是 README 快速开始/示例配置的 CLI 选择。5ep 配置本身没有该字段，实际值由构建缓存时的 CLI 决定并写入 `index.json` | `moss_lora/cache.py:57, 537, 708`；`README.md:93`；`configs/train_5090_v4_5ep_r64_bs8.yaml`（无该字段）`[未验证]` |
| 9 | tokenizer 是 **right truncation**，chat template 前缀只有 3 token，artist 在正常 caption 的第 7–9 个 token（0-based），512 截断不会切掉 artist；但 layout JSON 的 artist 位置取决于 JSON key 顺序 | `moss_lora/text.py:31-35`；本机 tokenizer `[实测]`；`moss_lora/annotations.py:587-593` |
| 10 | 代码库没有任何 A1111 风格 `(artist:foo:1.3)` 加权解析；该字符串会被当普通 tag/prose 编码。tokenizer 会把 `(artist` 合成一个不同的 token，artist 关键词本身反而被破坏 | `scripts/prepare_captions.py:313`；`moss_lora/annotations.py:88-124`；`[实测]` |
| 11 | `caption_no_artist` 只写入 metadata，**不缓存**；训练循环没有任何 text/artist dropout，也没有 CFG | `docs/CAPTIONS.md:233`；`moss_lora/cache.py:59`；`moss_lora/trainer.py:456-470` |
| 12 | 默认 LoRA 只打在 42 层 DiT 的 `q,k,v,o,ffn_up_gate,ffn_down`；`text_projection=false`、`final_linear=false`，文本投影和 2 个 text block 全部冻结 | `configs/train_5090_v4_5ep_r64_bs8.yaml`；`moss_lora/lora.py:101-111` |
| 13 | `text_projection=true` 只给 `text_embedder.proj` 加 LoRA，且 **ComfyUI 导出会失败**（导出器只接受 `layers.N.proj.*`） | `moss_lora/export_comfy.py:32, 90-107`；`docs/COMFYUI.md:80-100` |
| 14 | 缓存的 text 张量不进入导出的 LoRA 文件：导出物只有 DiT 权重。任何“缓存期改 embedding”的训练/推理一致性都必须由 prompt 文本或外部代码复现，ComfyUI 工作流本身无法携带 | `moss_lora/export_comfy.py:1-16`；`docs/COMFYUI.md:96-110` |
| 15 | 缩放 artist span 的乘性尺度大部分被逐 token RMSNorm 抵消；存活的是残差幅度 + 线性 bias 引起的方向扰动。真实权重实测：`s=1.3` 时 artist-key attention mass 平均仅 +0.00055，且不同 head 正负不一、随 `s` 非单调 | `moss_lora/model/moss_image.py:88-95, 125-142, 170-180`；本文 §5.3 `[实测]` |

---

## 1. DiT 到底怎么“看到”文本

### 1.1 文本编码器

`moss_lora/text.py:21-38`：

```python
@torch.no_grad()
def encode(self, prompt, max_length=None):
    formatted = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True,
                            max_length=max_length or self.max_length)
    inputs = {key: value.to(self.device) for key, value in inputs.items()}
    output = self.model(**inputs, use_cache=False, return_dict=True)
    return output.last_hidden_state[0]
```

- 模型通过 `AutoModel.from_pretrained(...)` 加载（`text.py:51-64`），`eval().requires_grad_(False)`，冻结。
- 返回的是 **最后一层逐 token hidden state** `[L, 4096]`，没有 pooling、没有 CLIP pooled vector、没有 `last_hidden_state.mean(...)`。
- `Qwen3_5Model` 是 decoder-only / causal 结构：本机 config 显示 32 层，其中 24 层 `linear_attention`、8 层 `full_attention`（每 4 层一次 full attention）。`use_cache=False` 只关 KV cache，不改变 causal 方向。
- chat template 是固定的：`<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n<think>\n`（`/mnt/data/MOSS-Image-Base/tokenizer/chat_template.jinja`）。`add_generation_prompt=True` + `enable_thinking=True` 意味着 **每条缓存 embedding 都包含末尾 7 个 assistant/think token**；它们也参与 DiT 的条件序列。

### 1.2 DiT 的文本路径

`moss_lora/model/moss_image.py` 的关键顺序：

```python
# 258-270: 校验 [B, S, text_dim] 和 binary mask
# 282-291: 逐样本处理
context = encoder_hidden_states[index:index + 1, mask[index].bool()]  # 283
video = self.video_embedder(...)                                      # 286-287
rope = self.rope(height, width, context.shape[1], ...)                # 289
text = self.text_embedder(context, rope[:, height * width:])          # 290
x = torch.cat((video, text), 1)                                       # 291
for layer_index, layer in enumerate(self.layers):                     # 292
    x = layer(x, rope, mod[index:index + 1])                          # 298
x = self.final_norm_video(x[:, :height * width])                      # 299
```

- `text_embedder` 是 `_TextProjection`：`nn.Linear(4096, 3072)` + **2 个 `_TextBlock`**（`moss_image.py:145-155`）。这 2 个 block 只对文本 token 做 self-attention，**没有 causal mask**；因此进入 joint block 前，artist token 已经可以和后面的 prose/tag token 双向混合。
- 之后 `torch.cat((video, text), 1)`，42 个 `_Block` 对 image + text 一起做 **joint self-attention**。这不是 cross-attention，也没有独立的 text K/V 投影；文本 token 和 image token 共享同一套 `q/k/v/o`。
- `final_norm_video` 只取前 `height*width` 个 token（`moss_image.py:299`），所以 **text token 不参与 loss**；`docs/RESEARCH.md:109` 也明确写了“文本 token 参与 joint attention，但不参与 loss”。
- `encoder_attention_mask` 的实际作用：每个样本按 `mask[index].bool()` 取真实 token（`moss_image.py:283`），padding 被完全丢弃。由于 batch 是逐样本循环处理的，padding **不会**通过 attention 影响结果；这也意味着 mask 不能表达“token 权重”，只能表达“哪些位置有效”。
- `skip_layers` 是整层跳过（`moss_image.py:260, 292-298`），只用于推理时的 negative/SLG 分支；训练不传 skip_layers，也没有 CFG（`trainer.py:467-468` 只有一次 forward）。
- 推理时 CFG 是两次 forward：positive + negative，`prediction = uncond + guidance_scale * (prediction - uncond)`（`moss_image.py:460-465`）；`run_validation` 使用缓存好的 negative embedding、`sample_cfg=3.0`、`sample_skip_layer=22`（`trainer.py:337-353`；配置 `logging.*`）。

**机械结论**：artist 条件是否“强”，取决于 DiT 的 joint attention 在 42 层里给 artist token 分配多少注意力，以及 LoRA 是否学会了读取它；缓存只决定 artist token 的初始向量，不决定注意力权重。

---

## 2. 缓存里到底存了什么

### 2.1 每个 item 的文件

`build_cache` 的 phase 1（`cache.py:629-640`）：

```python
embeddings = [encoder.encode(variant, max_length=max_text_length or 4096)...]
embeddings = [embedding.to(torch.bfloat16) for embedding in embeddings]
text_dim = int(embeddings[0].shape[1])
tensors = {f"text_{position}": e for position, e in enumerate(embeddings)}
tensors["text_lengths"] = torch.tensor([e.shape[0] for e in embeddings], dtype=torch.int64)
```

phase 2 再把 `latent` 合并进同一个 item 文件（`cache.py:686-695`）。

| 文件 | key | shape / dtype | 说明 |
| --- | --- | --- | --- |
| `item_XXXXX.text.safetensors`（phase 1 中间文件） | `text_0..text_{K-1}` | `[L_i, 4096]`，bf16 | 每个 variant 的完整逐 token hidden state，**无 padding** |
| 同上 | `text_lengths` | `[K]`，int64 | 每个 variant 的真实 token 数 `L_i` |
| `item_XXXXX.safetensors`（最终） | `latent` | `[128, h, w]`，fp32 或 bf16 | VAE latent |
| 同上 | `text_*` / `text_lengths` | 同上 | 从 phase 1 搬过来 |
| `negative.safetensors` | `embedding` | `[L_neg, 4096]`，bf16 | 负向 prompt 的逐 token hidden state |
| 同上 | `length` | scalar int64 | 负向长度 |
| `index.json` | — | JSON | `text_dim`、`max_text_length`、`variant_weight_overrides`、每 item 的 `variant_kinds` / `variant_weights` / `text_lengths` / hash / bucket |

**没有**：pooled embedding、attention mask、per-token weight、token id、文本原文（原文只在 metadata/index 的 hash 中间接存在）。`text_lengths` 只用于索引/校验；训练时 `collate_batch` 是根据张量形状重新造 mask 的（`dataset.py:186-192`）。

### 2.2 缓存复用条件

`_file_matches`（`cache.py:471-490`）只比较：

- `image_sha256`
- `caption_sha256` 列表（每个 variant 的文本 hash）
- `revision`
- `max_text_length`

它 **不校验张量值**。因此手工修改 `.safetensors` 里的 `text_i` 不会被缓存复用逻辑发现；但只要 caption 文本改变或重建缓存，就会被覆盖。`_negative_matches` 同理（`cache.py:493-510`）。

---

## 3. Variant 权重与期望混合

### 3.1 默认权重

`cache.py:61-67`：

$$
w_{\text{original}}=0.16,\quad
w_{\text{long}}=w_{\text{medium}}=w_{\text{short}}=\frac{0.64}{3}\approx0.21333,\quad
w_{\text{layout}}=0.20
$$

总和为 1。

每个 item 只保留实际存在的 kind，然后重新归一化（`normalize_variant_weights`，`cache.py:78-101`）：

$$
\tilde w_k=\frac{w_k}{\sum_{j\in K_{\text{item}}}w_j}
$$

`pick_variant` 用 `random.choices(..., weights=weights)` 抽样（`dataset.py:51-63`）；`data.variant_weights` 覆盖时，不在 override 里的 kind 权重为 0；全 0 时回退均匀。

### 3.2 两种典型 item

**有全部 5 个 variant：**

| variant | 权重 | 百分比 |
| --- | --- | --- |
| original | 0.16 | 16.00% |
| long | 0.64/3 | 21.33% |
| medium | 0.64/3 | 21.33% |
| short | 0.64/3 | 21.33% |
| layout | 0.20 | 20.00% |

**没有 layout（4 个 variant）：** 分母变为 `0.16 + 0.64 = 0.80`：

| variant | 权重 | 百分比 |
| --- | --- | --- |
| original | 0.16/0.80 = 0.20 | 20.00% |
| long | (0.64/3)/0.80 = 0.26667 | 26.67% |
| medium | 0.26667 | 26.67% |
| short | 0.26667 | 26.67% |
| layout | 0 | 0% |

### 3.3 与观测直方图对比

设 `p` 为“有 layout”的 item 比例，并假设其余 item 有 original + 3 个 prose（没有 layout）。则总体期望：

$$
E[\text{layout}]=0.20p
$$

$$
E[\text{original}]=0.16p+0.20(1-p)=0.20-0.04p
$$

$$
E[\text{each prose}]=0.21333p+0.26667(1-p)=0.26667-0.05333p
$$

观测值来自 `docs/bs8_5ep_status.md:84`（13,120 个样本）：short 23.1%、medium 23.2%、long 22.7%、original 17.7%、layout 13.2%。

由 `0.20p = 0.132` 得：

$$
p \approx 0.66
$$

代回：

| variant | 全部 5 个的期望 | 没有 layout 的期望 | `p=0.66` 混合期望 | 观测 |
| --- | --- | --- | --- | --- |
| original | 16.00% | 20.00% | **17.36%** | 17.7% |
| long | 21.33% | 26.67% | **23.15%** | 22.7% |
| medium | 21.33% | 26.67% | **23.15%** | 23.2% |
| short | 21.33% | 26.67% | **23.15%** | 23.1% |
| layout | 20.00% | 0% | **13.20%** | 13.2% |

`[推断]` 观测与 `p≈2/3` 的混合比例吻合（original 差 0.34 pp，约 1σ；prose 在 0.1–0.45 pp 内），而不符合“所有 item 都有 5 个 variant”（那会得到 layout 20%、original 16%）。因此该缓存里大约 **1/3 的 item 没有 layout**（可能是没有 OCR/layout 的图片），缺失 layout 的 item 在采样时会自动把 layout 的 20% 分给其余 4 个 variant。实际 `index.json` 不可访问（`/home/user/cache/clean30_v4_1m` 在本工作区不存在），无法直接统计 `variant_kinds` 分布；`[未验证]`。

> 文档不一致：`docs/CAPTIONS.md:233` 仍写“long/medium/short 中均匀选一个”，`docs/CAPTIONS.md:384` 写 layout “约 1/3 的 step”。前者与 `cache.py:61-67` 的加权默认值不符；后者是 caption 预处理阶段的假设，不是 cache 采样权重。以代码和 `docs/bs8_5ep_status.md:84` 为准。

---

## 4. 截断（`max_text_length`）与 mask

### 4.1 实际限制来自哪里

- `cache.py:57`：`DEFAULT_MAX_TEXT_LENGTH = 1024`。
- `cache.py:58`：`MAX_TEXT_LENGTH = 4096`。
- CLI `--max-text-length` 默认 1024（`cache.py:787-789`）。
- `README.md:93` 的快速开始用 `--max-text-length 512`；`docs/RESEARCH.md:109` 说“示例配置用 512 控制成本”。
- 5ep 配置文件 `configs/train_5090_v4_5ep_r64_bs8.yaml` **没有** `data.max_text_length` 字段；训练器也不读这个值。实际缓存是构建时 CLI 决定的，并写入 `index.json` 的 `max_text_length`（`cache.py:708`）。`[未验证]` 5ep 缓存到底是不是 512；题设按 512 分析。

### 4.2 截断位置

`text.py:34-35` 调用 `tokenizer(..., truncation=True, max_length=N)`。本机 tokenizer 的 `truncation_side='right'`、`padding_side='right'`，所以是 **从右侧截断**。

用 canonical caption 实测（`[实测]`）：

```
caption = "mystyle, artist:alice, character:hero, landscape, sunset, city lights, A city street at dusk."
formatted = "<|im_start|>user\n" + caption + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
总 token 数 = 35
前缀 = 3（"<|im_start|>", "user", "\n"）
caption = 25
后缀 = 7（"<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n"）
artist 的 3 个 token 位置（0-based）= [7, 8, 9]（" artist", ":", "alice"）
```

因此内容预算为：

$$
L_{\text{content}} = N_{\max} - 3 - 7 = N_{\max} - 10
$$

- `N_max = 512` → 内容 502 token；
- `N_max = 1024` → 内容 1014 token。

artist 在正常 caption 的开头（trigger 之后），位置约 7–9，只要 `N_max >= 10` 就不会被截掉。`[实测]` 把同一条 caption 用 `max_length=8` 截断仍保留 `" artist"`，`max_length=6` 才只剩 `"mystyle"`。

### 4.3 layout 的特殊风险

layout variant 是紧凑 JSON，不是自然语言。生成的 `Layout.to_dict()` 把 `summary` 放在第一位（`moss_lora/annotations.py:587-593`）：

```python
{
  "summary": {"meta": [...], "canvas": ..., "caption": "<含 artist 的 caption>"},
  "items": [...],
  "bbox_coord_space": ...,
  "bbox_format": ...
}
```

`_layout_text` 用 `json.dumps(dict(value), ...)` 保留输入的 key 顺序（`cache.py:180-199`）。如果上游 JSON 把 `items` 放在 `summary` 前面，artist 会出现在所有 item 之后，512 截断就可能把它切掉；测试里也接受这种重排（`tests/test_cache_variants.py:75-85`）。生成的 layout 是 summary 在前，所以常规路径 artist 安全。

用本机 tokenizer 对合成 layout 实测（`[实测]`，实际文本长度会因 item 内容而变）：

| layout item 数 | 总 token 数（含 template） | 512 截断后 |
| --- | --- | --- |
| 5 | 217 | 完整 |
| 20 | 687 | 尾部 item 被截 |
| 50 | 1,307 | 大量 item 被截 |
| 100 | 3,167 | 大量 item 被截 |

20-item layout 的 artist span 是 `[20, 21, 22]`（summary 在前），512 截断仍保留；但如果 key 顺序是 items 在前，就可能被切。`docs/CAPTIONS.md:390-397` 的 layout token 预算是在 **预处理阶段** 用 `--layout-max-tokens`（默认 1024）控制的，不是 cache 的 `--max-text-length`；两者可能不一致。

### 4.4 是否有 mask / 截断标记

- 缓存文件里 **没有 mask**；`text_i` 本身是未 padding 的 `[L_i, D]`。
- `collate_batch` 在 batch 内右侧 padding 到 `Lmax`，并生成 bool mask（`dataset.py:177-195`）。
- DiT 用 mask 逐样本取真实 token（`moss_image.py:283`），所以 padding 不影响结果；**mask 不是 attention bias，不能做加权，也不能恢复被截断的 token**。
- 缓存 **不记录“是否发生截断”**。唯一线索是 `text_lengths` 里某个值恰好等于 `max_text_length`；但这也可能是自然长度正好等于上限。`_file_matches` 只检查 `max_text_length` 元数据和长度 > 0（`cache.py:471-510`），不会报警。
- 截断是静默的：cache 构建日志不打印每个 item 的截断情况；`docs/CAPTIONS.md:483` 的 token 报告是 `prepare_captions` 阶段生成的，不是 cache 阶段。

### 4.5 训练/推理一致性

- 训练用缓存 embedding；推理 `sample.py` 默认 `--max-text-length 1024`（`sample.py:103-104`），`MOSSImagePipeline.encode_prompt` 默认 `max_sequence_length=4096`（`moss_image.py:361-383`）。
- 如果缓存是 512，而推理用 1024 编码长 prompt，超过 512 的部分在训练中从未出现；DiT 能接受任意长度（`[B,S,4096]`，无固定 shape 要求），但条件分布不一致。
- ComfyUI 的 MOSS 文本编码节点使用自己的 tokenizer/长度设置；本仓库没有它的实现。`[推断]` 需要把 ComfyUI 的 max token 设成与缓存一致，否则同样有分布偏移。

---

## 5. “缓存时加强 artist”的四种方案评估

> 先强调：缓存阶段没有“权重”参数。`build_cache` 只接收 `variant_weights`（选择哪条文本），不接收 token 权重、不做 embedding 后处理。`make_text_encoder` 返回的是冻结 `TextEncoder`，`encode()` 只做 tokenize → forward → `last_hidden_state`（`cache.py:757-766`；`text.py:29-38`）。所以任何“加强”都必须落在以下四类之一。

### 5.1 方案对比

| 方案 | 当前代码是否支持 | 机制上发生什么 | 与推理期加权等价？ | 对风格绑定可能的作用 | 主要风险 |
| --- | --- | --- | --- | --- | --- |
| (a) 在 caption 中重复 artist tag | **支持**（cache 接受任意 caption 文本；但 `prepare_captions` 会去重重复 tag） | 增加若干 artist-bearing token/key；causal TE 中后出现的副本会看到前面的 prose，表示被上下文混合 | 否；只是“更多 key”，不是乘性权重 | 弱；可能略微增加注意力份额，不保证 | 占 token 预算；改变分布；推理必须同样重复；可能过拟合重复模式 |
| (b) 编码后缩放 artist span 的 hidden states | **不支持**（无 hook；只能外部改 `.safetensors` 或改代码） | 逐 token RMSNorm 会消掉乘性尺度；存活的是残差流幅度和线性 bias 引起的方向变化；q/k 不会得到 `log(w)` 增益 | 否；A1111 是在 CLIP 输入空间做乘性权重，这里是在冻结 TE 之后、RMSNorm 之前 | **大部分抵消，非严格 no-op**；剩余是小的、head-dependent 扰动，不是可控强调 | 训练/推理不一致；bf16 精度；重建缓存会覆盖；ComfyUI 无法复现 |
| (c) 学习 per-token weight，供 DiT attention 消费 | **不支持**（无字段、无 attention bias、无 cross-attention） | 需要在 joint attention 的 logits 上加 `log(w)` 或对 v 加权；必须端到端训练 | 只有训练并推理都传同一权重时才可能等价 | 理论上是唯一能实现“真强调”的机制；但属于研究/架构改动 | 改模型、改缓存 schema、改 collate、改导出；未训练的权重无效；可能破坏预训练条件 |
| (d) 训练 text projection adapter | **部分支持**：`lora.text_projection=true` 只给 `text_embedder.proj` 加 LoRA | 全局重标定 4096→3072 的文本投影；2 个 text block 仍冻结；不是 per-token | 否；全局适配器无法只放大某个 artist token | 可能改善 TE→DiT 接口对齐；对“某个 artist”绑定作用有限 | **ComfyUI 导出失败**；额外参数；训练/部署路径不一致 |

### 5.2 (a) 重复 artist tag：为什么不是等价于权重

- `prepare_captions` 的 `dedupe_tags` 会按规范化 key 去重（`annotations.py:101-124`），`compose_caption` 的 prefix 只放一次 `artist:<id>`（`annotations.py:1043-1057, 1112-1115`）。要重复必须直接改 metadata/caption 文本，绕过预处理。
- 在 causal TE 中，第一个 `artist` token 只看到 trigger 和它前面的内容；第二个 `artist` token 会看到第一个以及中间 prose。它不是第一个 token 的简单复制，而是“artist + 场景”的混合表示。
- joint attention 的 softmax 会让多个相似 key 分摊注意力；增加副本不等于把权重乘以 1.3，效果取决于训练时是否见过这种重复模式。
- 推理时必须重复同样的文本；否则训练/推理分布不一致。对 ComfyUI 用户来说这可以做到（就是 prompt 里重复），但会消耗 token 预算，且可能让 caption 变得不自然。

### 5.3 (b) 缩放 hidden states：RMSNorm 到底抵消了多少？

**结论先说：这不是“完全 no-op”，但也绝不是“把 artist 权重乘以 1.3”。** 逐 token RMSNorm 会消掉乘性尺度；存活下来的是 **残差流里的幅度** 和 **线性层 bias / 非线性导致的少量方向变化**。这些剩余量会改变文本表示，但不会在 attention logits 上给 artist token 一个可控的加法/乘法增益。

#### 5.3.1 精确的调用链

`_TextProjection`（`moss_image.py:145-155`）：

```python
self.proj = nn.Linear(text_dim, dim)          # 4096 -> 3072, bias=True
self.blocks = ModuleList([_TextBlock(...) for _ in range(2)])
def forward(self, x, rope):
    x = self.proj(x.to(self.proj.weight.dtype))
    for block in self.blocks:
        x = block(x, rope)
    return x
```

`_TextBlock`（`moss_image.py:125-142`）的第一 op 确实是 `norm1`：

```python
q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
...
x = x + self.proj(_attention(q, k, v, rope, self.heads))
y = self.norm2(x)
return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))
```

`_Block`（joint block，`moss_image.py:170-180`）同样先 `norm1`：

```python
y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
q = self.q_norm(self.proj['q'](y)...)
k = self.k_norm(self.proj['k'](y)...)
x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), ...))
```

`_RMSNorm`（`moss_image.py:88-95`）是标准的逐 token RMSNorm：

```python
y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
return y.to(x.dtype) * self.weight
```

对任意 `s > 0`，忽略 eps：

$$
\operatorname{RMSNorm}(s\cdot x)=\operatorname{RMSNorm}(x)
$$

也就是说，**如果 norm 的输入正好是 `s·x`，它的输出完全不变**；q/k/v 也不变，attention 权重不变。这就是“RMSNorm 抵消缩放”的严格部分。

#### 5.3.2 那为什么不是严格 no-op？什么存活了？

有三个通道让缩放“漏”过去：

1. **投影 bias（第一个通道）**：`_TextProjection.proj` 是 `nn.Linear(4096, 3072)`，默认 `bias=True`。缩放后的输入满足
   $$
   \operatorname{proj}(s\cdot x)=s\cdot Wx+b \neq s\cdot(Wx+b)
   $$
   所以 `norm1` 看到的不是 `s·(Wx+b)`，而是 `s·Wx+b`；bias 与信号的比例变了，归一化后的方向就变了。本机 checkpoint 的 `text_embedder.proj.bias` 确实存在，norm ≈ 8.03（`/mnt/data/MOSS-Image-Base/transformer`，`[实测]`）。
2. **残差流（主通道）**：`x = x + f(norm(x))`。norm 只作用在分支输入上；**残差 `x` 本身不会被归一化**。缩放 artist 行后，残差里带着 `s·x`，而该 block 的 update `f(...)` 大致不变；下一层 `norm1` 看到的是“放大的旧残差 + 未放大的 update”的混合，方向会改变。这个效应会跨 2 个 text block 累积，然后进入 joint block。
3. **非线性 + 多层累积**：`_situ`/SiLU、attention 的 softmax、后续 block 的 `norm2` 都会把上面的方向变化进一步混合；到 joint block 的 `norm1` 时，文本 token 的 q/k/v 已经和 baseline 不同。

**关键点**：这三个通道都是“改变文本向量的方向/内容”，不是“给 artist token 的 attention logits 加一个 `log(w)`”。逐 token RMSNorm 是 **per-token** 的：它不会因为 artist token 的范数更大，就让它在 softmax 里压过其他 token。q/k 是在每个 token 内部归一化后算的；缩放不会变成 q/k 的幅度增益。

#### 5.3.3 用真实预训练权重做的量化验证

用本机 `/mnt/data/MOSS-Image-Base/transformer` 的真实 `text_embedder.*` 和 `layers.0.*` 权重（float32 CPU），输入 8 个 text token（每 token RMS=1，norm=64）+ 4 个随机 video token；把 text 第 3–5 行（模拟 artist span）乘以 `s`，比较：

| `s` | `_TextProjection` 输出相对 L2 差 | joint `layers.0` q/k/v 相对差 | 视频 query 对 artist key 的 attention mass 平均 Δ | 每 head Δ 范围 |
| --- | --- | --- | --- | --- |
| 1.3 | 0.0955 | 0.0599 / 0.0658 / 0.0665 | **+0.00055** | [−0.00985, +0.01455] |
| 2.0 | 0.2609 | 0.1556 / 0.1669 / 0.1685 | **−0.00017** | [−0.03377, +0.02395] |
| 3.0 | 0.4437 | 0.2387 / 0.2488 / 0.2530 | **−0.00106** | [−0.05671, +0.02638] |
| 10.0 | 1.5877 | 0.4132 / 0.4087 / 0.4183 | **+0.00263** | [−0.08564, +0.11634] |

基线 artist key 的 attention mass 约 0.06–0.26/head（平均约 0.15）。`s=1.3` 时平均变化约 **+0.00055（≈0.4% 相对）**，而且 **不同 head 有正有负**；`s` 增大时平均 Δ 不单调（1.3 为正、2/3 为负、10 又为正），只是扰动幅度变大。这不是“把 artist 权重调大”的行为。

补充：`s=1.3` 时第一个 `_TextBlock` 的 qkv 相对差只有约 **0.0060**（RMSNorm 在单层内几乎完全抵消）；到 joint `layers.0` 的 qkv 差约 **0.06**，主要是残差流跨 2 个 text block 累积造成的（`[推断]`，基于第一层 qkv 几乎不变、而最终文本表示变化 0.0955 的对比）。也就是说，**单层 norm 抵消得很干净，跨层残差又把一部分“漏”回来**。

> 实验是合成输入（unit-RMS hidden state + 随机 video token），不是真实 Qwen 输出；数值只用于说明机制和量级，真实模型的绝对数值会随 hidden state 分布变化。但“q/k 没有乘性增益、attention mass 变化小且 head-dependent、非单调”这一结论来自结构，不依赖具体数值。`[实测/推断]`

#### 5.3.4 为什么它不是“增加 attention mass”

要让 DiT 更“听” artist，需要改变 attention 分布本身，例如：

$$
\text{logits}_{ij}=\frac{q_i k_j^\top}{\sqrt d}+b_j,\quad b_j=\log w_j
$$

或对 artist token 的 value 做 `v_j *= w_j`。当前 `_attention`（`moss_image.py:117-122`）只做 `F.scaled_dot_product_attention(q,k,v)`，**没有 mask/bias/weight 参数**。缩放 hidden state 不会进入这个公式的 `b_j` 或 `v_j` 乘法；它只改变 `q/k/v` 的输入方向，而每个 token 的 q/k 又先被 RMSNorm 归一化。因此：

- 它不是 A1111 的 `(artist:foo:1.3)`：A1111 在 CLIP token embedding/attention 层做乘性权重；这里是在冻结 32 层 causal TE **之后**、DiT 的 RMSNorm **之前**做缩放，作用点完全不同。
- 它也不是“训练时给 artist 更高 loss 权重”：loss 只对 image token 计算（`moss_image.py:299`），文本 token 没有 per-token loss 权重。
- 它更不是“cache 里存了权重”：cache 没有这个字段（见 §5.6）。

#### 5.3.5 其他现实问题

- 缓存是 bf16；缩放后再存会引入额外量化。
- `_file_matches` 不校验张量值（`cache.py:471-490`），所以手工改张量不会被发现；但重建缓存会覆盖。
- `sample.py`/ComfyUI 不会自动做同样的缩放；除非你在推理时也改 `prompt_embeds`，否则训练/推理条件不一致。导出的 LoRA 只含 DiT 权重（`export_comfy.py`），不携带任何 text 处理逻辑。
- 大倍数（如 10）会让文本表示远离流形，可能破坏条件而不是“加强”它。

**判定：方案 (b) 的乘性尺度大部分被 RMSNorm 抵消；存活的是残差幅度和 bias 引起的方向扰动，效果小、head-dependent、非单调，不能当作 artist 加权。它是“不可控扰动”，不是“缓存期强调”。**

### 5.4 (c) 学习 per-token weight：唯一“真强调”，但不是缓存技巧

当前模型没有 cross-attention，因此不能“给 cross-attention 的 K/V 加权”。可行实现是在 joint attention 的 logits 上对 text key 加 bias：

$$
\text{logits}_{ij} = \frac{q_i k_j^\top}{\sqrt{d}} + b_j,\quad b_j=\log w_j \text{ for text token } j
$$

或者对 text token 的 `v` 做缩放。两种都需要：

1. `cache.py` 存 `text_weights`（或从 caption 解析出 span）；
2. `dataset.collate_batch` padding 时同步 padding 权重；
3. `MOSSImageTransformer.forward` 接收权重并传给 `_attention`（当前 `_attention` 没有 mask/bias 参数，`moss_image.py:112-122`）；
4. `trainer.py` 把权重传进 forward；
5. 推理 pipeline / sample.py / ComfyUI 也要传同样的权重；
6. 必须训练，否则权重只是未校准的扰动。

这不是“缓存时加强”，而是新的条件接口。若只是把权重设成 1.3 而不训练，模型没有学过这个维度的含义，效果不可预期（可能被忽略，也可能破坏条件）。`[推断]`

### 5.5 (d) text projection adapter：全局重标定，不是 per-artist

`lora.py:101-111`：

```python
paths = [f"layers.{index}.proj.{target}" for index in range(num_layers) for target in targets]
if text_projection:
    paths.append("text_embedder.proj")
if final_linear:
    paths.append("final_linear_video")
```

- `text_projection=true` 只包 `text_embedder.proj`（4096→3072 线性），**不包** `text_embedder.blocks`（2 个 text block 仍冻结）。
- 它学习的是“所有文本 token 共享”的投影校正，不能只针对 `artist:alice`。如果 TE 输出与 DiT 期望的尺度/方向不匹配，它可能有用；但它不是 artist 绑定机制。
- `export_comfy.py:32` 的 `KEY_PATTERN` 只匹配 `layers.N.proj.<target>.lora_A/B`；`text_embedder.proj` 的 key 会进入 `unexpected` 并抛 `ConversionError`（`export_comfy.py:90-107`）。`docs/COMFYUI.md:80-100` 也明确说该 adapter 没有 text-encoder tensor、不要把 LoRA 接到 CLIP。因此 `text_projection=true` 目前是 **训练/`sample.py` 实验**，不是 ComfyUI 部署路径。

### 5.6 运行时验证：cache 没有 per-token weight/mask，`variant_weights` 只选整条 caption

用 `tests.helpers.make_tiny_cache` 在临时目录构建一个 4-item 缓存，再用 `safe_open` 列出所有 tensor key（`[实测]`）：

| 文件 | tensors |
| --- | --- |
| `item_00001.safetensors`（3 个 variant） | `latent`, `text_0`, `text_1`, `text_2`, `text_lengths` |
| `negative.safetensors` | `embedding`, `length` |
| `index.json` 的 item | `id`, `image`, `image_sha256`, `caption_sha256`, `bucket_h`, `bucket_w`, `variant_kinds`, `variant_weights`, `text_lengths` |

- **没有** `text_weights`、`token_weights`、`mask`、`attention_mask`、`pooled` 之类的 key。`text_i` 是整条 `[L_i, D]` 序列；`text_lengths` 只是长度。
- `CachedDataset.__getitem__` 只返回一个 `text` 张量（`dataset.py:96-99`），不是“每个 token 一个权重”。
- `variant_weights` 的运行时行为：item 有 `long/medium/short` 时，默认权重经 `normalize_variant_weights` 变成各 1/3；`override={'short':1.0}` 时 `pick_variant` 只选 `short`。它选择的是 **整个 `text_i`**，不存在“对某条 caption 的某些 token 加权”的路径（`dataset.py:51-63`）。
- `collate_batch` 的 `mask` 是 `[B, Lmax]` bool，只标记“真实 token vs padding”（`dataset.py:177-195`）；DiT 用它 slice 掉 padding（`moss_image.py:283`），没有权重语义。

---

## 6. A1111 风格 `(artist:foo:1.3)` 支持吗？

**不支持，且会被当成普通文本，通常有害。**

- 全仓库没有 emphasis/weighting 解析器：`grep` 不到 `BREAK`、权重语法、`(token:1.3)` 解析；只有 `data.variant_weights`（variant 采样）、`lora` 的 `strength_model`、CFG 的 `guidance_scale`。
- `prepare_captions` 的 tag 解析（`scripts/prepare_captions.py:313`）用：

```python
match = re.match(r"(?i)^(artist|character)\s*:\s*(.*)$", token)
```

`(artist:alice:1.3)` 以 `(` 开头，匹配失败，被当作普通 tag 放进 `parsed.tags`。如果这是该图唯一的 artist 写法，`artist_missing=True`，最终 caption 还会补一个 `artist:unknown`（`docs/CAPTIONS.md:117-119`），于是同时出现 `artist:unknown` 和 `(artist:alice:1.3)`。
- `normalize_name` 只做 NFKC/空白/分隔符清理（`annotations.py:88-92`），不会去掉括号或 `:1.3`；`dedupe_tags` 也不会把它识别成 artist（`annotations.py:113-124`）。
- tokenizer 实测（`[实测]`）：

```
"artist:alice"          -> 3 tokens: ["artist", ":", "alice"]
"(artist:alice:1.3)"    -> 8 tokens: ["(artist", ":", "alice", ":", "1", ".", "3", ")"]
```

注意 `(artist` 被合并成一个 **不同的 token**（id 89682），而不是 `(` + `artist`（`artist` 的 id 是 18061）。也就是说，A1111 语法不仅没有加权，还替换掉了模型熟悉的 `artist` 关键词 token，并插入数字噪声。
- 推理端：`sample.py` 直接把 prompt 交给 `TextEncoder.encode`（`sample.py:113-118`），`MOSSImagePipeline.encode_prompt` 也只是套 chat template（`moss_image.py:361-383`）；ComfyUI 的 CLIP 路径同样没有本仓库的加权解析。`[推断]` ComfyUI 的 MOSS 自定义文本编码节点不会实现 A1111 权重；即使它有通用 CLIP 权重，也不适用于这里的 Qwen3.5 + 自定义 `TextEncoder`。
- 结论：`(artist:foo:1.3)` 在训练和推理中都是 **literal text**；最可能的结果是 artist 条件变弱/变噪，而不是变强。

---

## 7. `caption_no_artist` 与 dropout

### 7.1 当前状态

- `caption_no_artist` 由 `scripts/prepare_captions.py:1722` 写入 metadata；它是 long 变体去掉 `artist:<id>`、保留 character/tags/prose（`docs/CAPTIONS.md:139-140`）。
- cache **不缓存**它：`CAPTION_KEYS` 不含 `caption_no_artist`（`cache.py:59`），`_caption_records` 只解析 `caption_original` / `caption` / `text` / `prompt` / `image_description` / `long` / `medium` / `short` / `layout`（`cache.py:208-305`）。`docs/CAPTIONS.md:233` 明确写“当前 cache 不缓存它”。
- 训练循环没有任何 text dropout / artist dropout / CFG：只有一次 `model(...)` forward，loss 是纯 flow matching（`trainer.py:456-470`）。`lora.dropout` 只是 LoRA 层内的 dropout（`lora.py:40-63`），与条件无关。
- 验证时 negative 是构建缓存时固定的 `--negative-prompt`（`trainer.py:337-353`；`cache.py:642-649`），不是“去掉 artist 的同一 caption”。

### 7.2 要加入 artist dropout，需要做什么

1. **缓存层**：让 `caption_variant_records` 识别 `caption_no_artist`，产出 `CaptionVariant("no_artist", text)`；把它加入 `VARIANT_ORDER` 和 `DEFAULT_VARIANT_WEIGHTS`（或单独的 `--artist-dropout` 开关）。`normalize_variant_weights` 对未知 kind 默认 1.0（`cache.py:93-94`），但最好显式给权重，避免总和意外。重建缓存后 `caption_sha256` 列表变化，旧缓存自动失效（`cache.py:471-490`）。
2. **数据层**：两种实现：
   - 作为第 6 个 variant 参与 `pick_variant` 采样（改动最小，但“artist dropout”会和 caption 长度/格式耦合）；
   - 或让 `CachedDataset.__getitem__` 同时返回 `text` 和 `text_no_artist`，`collate_batch` 分别 padding，trainer 以概率 `p` 逐样本替换。后者更接近标准 CFG dropout，推荐。
3. **训练层**：`context = batch["text"]` 后按 mask/概率替换；loss 不需要改（仍然是去噪目标）。这样模型会学到 `p(x | no artist)`，为推理时的 artist-specific CFG 提供校准的无条件分支。
4. **推理层**：negative prompt 用同一 caption 的 `caption_no_artist` 文本，或至少用 `artist:unknown`。pipeline 已经接受 `negative_prompt_embeds`（`moss_image.py:387-445`），`sample.py` 可以用 `--negative-prompt`；ComfyUI 用负向文本即可，不需要改模型。
5. **成本/风险**：多一个 variant 的编码时间和磁盘；`p` 通常取 0.05–0.1，太大会削弱 artist 条件；需要足够训练步数让无 artist 分支收敛。`caption_no_artist` 只有 long 版本，若要对 medium/short 做 dropout，需要额外生成对应文本（或直接用 long 的 no-artist 作为统一无条件文本）。

> 注意：`artist:unknown` 是训练分布内的（缺失 artist 的图片会被写成 `artist:unknown`，`docs/CAPTIONS.md:117-119`），所以它比“完全空 prompt”更适合作为 artist CFG 的 negative；而 `caption_no_artist` 目前不在缓存里，不能直接用于训练 dropout。

---

## 8. LoRA 目标与文本侧可训练容量

5ep 配置：

```yaml
lora:
  rank: 64
  alpha: 64
  targets: [q, k, v, o, ffn_up_gate, ffn_down]
  text_projection: false
  final_linear: false
```

- 42 层 × 6 个 target = 252 个 `LoRALinear`；`alpha=rank` 所以缩放为 1.0（`lora.py:47-48`；`docs/COMFYUI.md:140-146`）。
- `text_embedder.proj`（4096→3072）和 `text_embedder.blocks`（2 个文本 self-attention block）默认 **冻结**；`final_linear_video` 也冻结。
- 但 `q/k/v/o` 是 **image + text 共享** 的 joint attention 投影；LoRA 在这些层上会同时改变文本 token 的 q/k/v/o。因此存在“隐式的文本适配”，只是它与 image 侧纠缠，不是 per-token、也不是 per-artist。
- **没有新的 token embedding 表**。如果推理时出现训练中从未见过的 `artist:<new_name>`，TE 会把它切成若干 subword 并给出一个 OOD 向量；LoRA 没有能力为这个新名字“创建”风格绑定。要绑定一个新 artist，必须把该名字和该 artist 的图片一起放进训练 caption；否则只能依赖全局 trigger（数据集里的 `@CH3OCH3` / `@mystyle`）表达风格。
- `text_projection=true` 只给 `text_embedder.proj` 加 rank-64 LoRA；**不覆盖** 2 个 text block。并且 `export_comfy.py` 不接受这个 key（`export_comfy.py:32, 90-107`），所以它不能进入当前的 ComfyUI 导出流程。`docs/CONFIG.md:115` 的默认建议也是“不动，优先保持文本条件稳定”。

**对“绑定新 artist token”的含义**：

- 单 artist 数据集：artist tag 是常量，风格实际上绑定在全局 trigger 上；加强 artist tag 没有信息增益。
- 多 artist 数据集：artist tag 是唯一的风格判别变量；模型能不能“听”它，取决于数据中 artist 与风格的对应是否干净、每个 artist 的样本是否足够，以及 LoRA 容量/步数，而不是缓存的数值大小。
- 新 artist 名字：必须先有该 artist 的训练图片和 caption；仅靠推理时写一个新名字，模型无法从名字推断风格。

---

## 9. 如果目标是“让 artist tag 主导风格”，正确干预点是什么

### 9.1 推荐顺序

1. **Caption 构造（最高杠杆）**
   - 保持 `artist:<id>` 在 trigger 之后、最前面（当前 `build_prefix` 已经是这样：`annotations.py:1043-1057`）。
   - 使用稳定、规范化的 artist id；不要把同一 artist 写成多个别名。
   - **去混淆**：删掉与 artist 强相关、会替代 artist 解释风格的 prose/tag（例如每个 artist 固定的“anime coloring”“soft shading”等）。如果 prose 已经能预测风格，模型没有动力去读 artist tag。
   - 每个 artist 至少保证足够样本（`docs/CAPTIONS.md:191-193` 提醒 1–2 张会让 artist 条件过拟合到构图而非画风）。
   - 不要用 A1111 语法；它会破坏 `artist` token。
2. **训练目标（次高杠杆）**
   - 加入 artist dropout（缓存 `caption_no_artist` + 逐样本替换），让模型学到“去掉 artist 后的分布”，从而在推理时用 CFG 放大 artist。这是把“artist 主导”变成可控生成行为的正确方式。
   - 若要更多文本侧适配，可试 `text_projection=true`，但要知道它只改 `text_embedder.proj`、且不能走当前 ComfyUI 导出；更适合作为 `sample.py` 实验。
   - 容量/步数：rank 64 已经不小；如果 loss 长期平坦（`docs/bs8_5ep_status.md:69-76`），优先检查数据去混淆和 artist 覆盖，而不是继续加 rank。
3. **推理（可控杠杆）**
   - 用 **不含 artist 的 negative prompt**（或 `artist:unknown`）做 CFG；这是当前代码唯一支持的“artist 强调”手段，因为 CFG 的负向分支可以是任意文本。
   - `guidance_scale` 不要盲目拉高；官方/配置默认 3.0（`docs/CONFIG.md:117`），过高会过饱和。
   - `strength_model` 保持 0.8–1.0（`docs/COMFYUI.md:90, 139`）；它缩放整个 LoRA，不是文本权重。
   - 确保推理 tokenizer、chat template、`max_text_length` 与缓存一致（`docs/FAQ.md:121`）。
4. **不要做的事**
   - 缓存张量缩放（大部分被 RMSNorm 抵消，剩余是不可控扰动）。
   - 在 caption 里堆重复 artist tag（弱、占预算、分布漂移）。
   - 用 `(artist:foo:1.3)`（不支持，破坏 token）。
   - 仅调 `data.variant_weights` 想“加强 artist”：所有已缓存 variant 都含 artist（除了未缓存的 `caption_no_artist`），variant 权重只改变 caption 形式比例，不改变 artist 的有无/强度。

### 9.2 判定表

| 操作 | 判定 | 原因 |
| --- | --- | --- |
| 缓存时缩放 artist span hidden states | **大部分抵消 / 不可控扰动（不推荐）** | RMSNorm 消掉乘性尺度；存活的是残差幅度 + bias 方向扰动；attention mass 变化小且 head-dependent、非单调；off-manifold；训练/推理不可复现；导出物不含文本处理 |
| 缓存时重复 artist tag | **弱，不推荐作为主手段** | 只是增加 key；causal 上下文混合；占 token；推理需同步重复 |
| A1111 `(artist:foo:1.3)` | **有害** | 无解析器；`(artist` 变成不同 token；可能触发 `artist:unknown` 回退 |
| 学习 per-token attention weight | **唯一真正的“强调”机制，但是架构改动** | 需改 attention logits、缓存 schema、collate、trainer、推理；必须训练 |
| `text_projection=true` | **训练实验可以，ComfyUI 不行** | 只改 `text_embedder.proj`；导出器拒绝该 key |
| 仅改 `data.variant_weights` / `layout_weight` | **对 artist 强度无效** | 只改 caption 形式采样比例；所有缓存 variant 都含 artist |
| artist dropout + CFG negative | **推荐** | 让模型学到无 artist 分支，推理时可放大 artist；需缓存 `caption_no_artist` |
| 数据去混淆 + 每 artist 足够样本 | **最高杠杆** | 让 artist 成为风格的唯一可靠预测变量，模型才会依赖它 |

---

## 10. 代码索引（按主题）

| 主题 | 位置 |
| --- | --- |
| 文本编码（chat template / truncation / last_hidden_state） | `moss_lora/text.py:29-38` |
| 冻结 TE 加载 | `moss_lora/text.py:21-26, 47-64` |
| 缓存默认长度 / 上限 | `moss_lora/cache.py:57-58` |
| variant 顺序 / 默认权重 | `moss_lora/cache.py:60-67` |
| 权重归一化 / override | `moss_lora/cache.py:78-101` |
| caption → variants（去重、layout 分离） | `moss_lora/cache.py:208-332` |
| 缓存写入（text + lengths） | `moss_lora/cache.py:629-634` |
| negative 缓存 | `moss_lora/cache.py:642-649` |
| 缓存复用校验（hash/revision/max length） | `moss_lora/cache.py:471-510` |
| 变体抽样 | `moss_lora/dataset.py:18-63` |
| collate padding + mask | `moss_lora/dataset.py:177-195` |
| 训练 forward（context/mask，无 dropout/CFG） | `moss_lora/trainer.py:456-470` |
| 验证 CFG / skip layer | `moss_lora/trainer.py:337-353` |
| DiT text projection（线性 + 2 blocks） | `moss_lora/model/moss_image.py:145-155` |
| DiT joint attention / 文本拼接 | `moss_lora/model/moss_image.py:283-299` |
| text block / block 的 RMSNorm | `moss_lora/model/moss_image.py:125-143, 170-180` |
| 推理 encode_prompt / CFG | `moss_lora/model/moss_image.py:361-383, 460-465` |
| LoRA 目标路径 / text_projection | `moss_lora/lora.py:101-111` |
| ComfyUI 导出 key 限制 | `moss_lora/export_comfy.py:32, 90-107` |
| sample.py 推理编码 | `moss_lora/sample.py:103-118` |
| layout 的 JSON key 顺序 | `moss_lora/annotations.py:587-593` |
| `caption_no_artist` 生成 | `scripts/prepare_captions.py:1722`；`docs/CAPTIONS.md:139-140, 233` |
| 观测 variant 直方图 | `docs/bs8_5ep_status.md:84` |
| 5ep LoRA 配置 | `configs/train_5090_v4_5ep_r64_bs8.yaml` |
| 文本 token 不参与 loss | `docs/RESEARCH.md:109` |

---

## 11. 未验证与推断清单

- `[未验证]` `configs/train_5090_v4_5ep_r64_bs8.yaml` 没有 `max_text_length` 字段；实际 5ep 缓存的 `max_text_length` 只能从 `/home/user/cache/clean30_v4_1m/index.json` 读取，本工作区该路径不存在。题设的 512 与 `README.md:93` / `docs/RESEARCH.md:109` 的示例一致，但代码默认是 1024。
- `[推断]` 观测 variant 直方图对应约 66% item 有 layout、34% 没有 layout；这是由 `p=0.132/0.20` 反推的，未直接统计 `index.json`。若存在 prose 变体去重（long==medium 等），实际 p 会略低；但去重会提高 original/layout 份额，无法单独解释 13.2% 的 layout。
- `[推断]` ComfyUI 的 MOSS 文本编码节点不会实现 A1111 权重；本仓库没有该节点实现，只能确认本仓库没有加权解析器。
- `[推断]` 方案 (c) 若只设权重不训练，效果不可预期；这是基于模型没有该输入维度的训练信号。
- `[实测]` token 位置/数量基于本机 `/mnt/data/MOSS-Image-Base/tokenizer` 和合成 caption/layout；实际数据集的 layout token 数以 `report.json` 的 `tokens` 统计为准（`docs/CAPTIONS.md:483`）。

---

## 12. 一句话答案

**“在缓存时加强 artist tag”在当前代码里不是一个有意义的旋钮**：缓存只是冻结 Qwen3.5 的逐 token hidden state；DiT 用 joint self-attention 读取整段文本，没有 per-token weight，也没有 cross-attention。改 caption 文本（重复 tag）只能弱影响且改变分布；改缓存张量（缩放 artist span）的乘性尺度会被逐 token RMSNorm 大部分抵消，剩下的是残差幅度/bias 方向扰动（attention mass 变化小、head-dependent、非单调），且无法在导出的 LoRA/ComfyUI 中复现；A1111 语法不支持且会破坏 `artist` token。要让 artist 主导风格，正确顺序是：**caption 去混淆 / 保证每 artist 样本 → artist dropout + CFG negative → 必要时 text_projection 实验 → 最后才考虑改 DiT attention 接口**。
