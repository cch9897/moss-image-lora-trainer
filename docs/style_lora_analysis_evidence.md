# MOSS-Image 风格 LoRA 失效：Caption 证据与根因排序

> 分析对象：`clean30_v4_5ep_r64_bs8`（导出 `step_010456`，4/5 epoch）。
> 分析方式：只读本地代码与本地数据集，不训练、不改代码；远程 `user@<other-host>` 一次重试仍为
> `No route to host`，因此**没有读取** `/home/user/cache/clean30_v4_1m/index.json`、`metrics.csv`、
> `samples/` 或完整 v4 metadata。凡是由本地样本/直方图外推的数字均标 `[推断]`。
>
> 证据标记：
> - `[实测]`：本地文件直接计算。
> - `[代码]`：仓库源码可直接查。
> - `[推断]`：由代码、配置、本地样本或训练直方图推导，未直接观测完整 v4 缓存。
> - `[未验证]`：需要远程文件才能确认。

---

## 0. 结论先行

1. **不是“artist tag token 太少”**。每个真实画师行的缓存变体里 artist tag 都存在，位置固定在 prefix 第 10 个
   token 左右，中位数 5 个 token，占文本 token 的 2.0%–7.7%（按采样权重加权约 4.2%）。它没有被截断，
   也没有被 layout 挤掉。`[实测 + 代码]`
2. **真正的问题是“每个 artist 的样本频率太低”**：255 个真实画师 + 771 条无画师行；画师图片数中位数
   74 张，只占 20,953 行的 **0.353%**。按 10,456 step、batch 8 计算，中位画师在整个 4 epoch 里出现在
   **约 292 个 batch（2.8% 的 step）**；尾部 28 张约 **111 个（1.07%）**；样本占比 0.353%/0.134%。
   而全局 trigger `@CH3OCH3` 出现在 **100%** 的样本里。`[实测 + 代码]`
3. **风格 prose 是捷径（次级放大器）**：按训练直方图与 finalize 代码推导，约 **86% 的 step 带 prose**
   （保守下界约 73%），只有约 **14%** 是 tag-only（且是 40 个 tag 的长 tag 行）。v4 finalize 的 `short`
   **也带一句 prose**，不是纯 tag 变体。`[推断 + 实测]`
4. **prose 的风格词是通用模板，不是画师判别信号**：本地 14,063 行扫描中 `digital` 44.0%、`cel shading`
   33.6%、`soft shading` 26.4%、`pastel` 16.0%，这些词命中的 top-5 画师占比只有 6–9%，而 top-5 画师本身
   约占 4.5% 图片——富集只有 1.3–2 倍。v4 pilot 里 `lighting` 38.2%、top-5 画师占 7.3%。`[实测]`
5. **“缓存时提高 artist 权重”在当前代码里不存在**。缓存保存的是冻结 Qwen3.5 的逐 token hidden state；
   没有 per-token weight，也没有 cross-attention 的 artist 权重。`variant_weights` 只决定采样哪个 caption
   变体，而所有真实画师行的变体都含 artist tag。唯一有意义的“加权”实验是**新增真正的 tag-only 变体并给
   它高权重**，不是调 `short=`。`[代码]`

---

## 1. 证据来源：本地真实数据 vs 远程未验证

| 文件 | 规模/内容 | 在本报告中的用途 | 可信度 |
|---|---|---|---|
| `/mnt/data/train_test/clean30_top256/metadata.jsonl` | **20,953 行**，`finalize_version=clean30-v1`，4,825 current + 16,128 expansion | 行集、画师分布、tag/layout 结构、v1 caption 构成 | `[实测]` 精确 |
| `/mnt/data/train_test/clean30_top256/report.json` | rows 20,953，255 个有画师条件，771 行无画师，15,114 行有 layout | 画师/数据构成 | `[实测]` |
| `/mnt/data/train_test/clean30_top256/DATASET_FORMAT.md` | schema、字段、variant 权重、行数 | 字段语义、权重、去重规则 | `[实测]` |
| `/mnt/data/train_test/ab_1k/final/metadata.jsonl` | **1,000 行**，`finalize_version=v2`，727 valid/repaired VLM + 273 fallback，810 layout | v4 finalize 路径的**真实样本**：prose 覆盖、token 长度、variant 构成 | `[实测]` 精确 |
| `/mnt/data/train_test/ab_1k/cache_1m/index.json` | 1,000-item cache，`max_text_length=1024` | 实际缓存 variant_kinds / variant_weights / text_lengths | `[实测]` 精确 |
| `/mnt/data/train_test/style_artist_strategy.md` | 设计文档，含 14,063 行 prose 风格词扫描 | 风格词通用性、top-5 画师富集、top-128/10% dropout 建议 | `[实测]`（本地扫描） |
| `/home/user/cache/clean30_v4_1m/index.json`、`metrics.csv`、`samples/` | **远程，SSH 失败** | 完整 v4 variant 覆盖、实际 `max_text_length`、后期 loss、样图 | `[未验证]` |

**关键限制，必须说清楚**：`clean30_top256/metadata.jsonl` 是 **v1 clean30 caption**，不是 v4 NSFW
recaption 后的完整 metadata；但它与 5ep 缓存共享**完全相同的 20,953 行图片集**（`report.json rows=20953`
与 `docs/bs8_5ep_launch.md:13` 的 cached items 20,953 一致）。因此：

- 画师计数、unknown 行数、layout 行数、tag 数量、图片集构成：**对 5ep 运行精确成立**。
- 每类 caption 的 prose 覆盖：v1 与 v4 不同；v4 用 `ab_1k/final` 的 1,000 行 v2 finalize 样本代表。
- 完整 v4 的 `long/medium/short` 覆盖：由训练日志直方图反推，标 `[推断]`。

### 1.1 为什么可以确定 5ep 缓存不是本地 v1 caption

本地 v1 metadata 经 cache 去重后，`long` 只在 2,628/20,953 行存在（12.5%）。如果 5ep 缓存用 v1
caption，`long` 的采样占比上限约 `0.125 × 0.267 ≈ 3.3%`，不可能达到
`docs/bs8_5ep_status.md:84` 观测的 **22.7%**。

反过来，用默认权重求长期望：有 layout 的 item 权重为
`original .16 / long .2133 / medium .2133 / short .2133 / layout .20`；无 layout 时按存活项归一化，
long/medium/short 各 `.2667`。若 `p` 为有 layout 的比例，则
`E[long] = .2133p + .2667(1-p) = .2667 - .0533p`。
观测 long=22.7% 要求 `p ≈ 0.75`；观测 layout=13.2% 要求 `p ≈ 0.66`。两者都只能在一个前提下成立：
**long/medium/short/original 几乎每个 item 都有**。因此 5ep 的 v4 缓存是“4 个文本变体 + 约 2/3 item
额外有 layout”，不是 v1 的“大量 item 没有 long”。`[推断，强]`

这也解释了 v4 pilot 与完整 v4 的差别：pilot 只有 727/1000 有 long（72.7%），完整 v4 的 long 覆盖
约 98%（否则 22.7% 无法达到）。v4 recaption 在完整集上的有效率远高于 1k pilot。`[推断]`

---

## 2. Caption 构造路径：代码级事实

### 2.1 前缀与变体

| 环节 | 文件:行 | 行为 |
|---|---|---|
| 前缀顺序 | `moss_lora/annotations.py:1035-1060` | `@trigger, artist:<id>, character:<id...>`；artist 在 trigger 之后、character/tags/prose 之前 |
| 拼接顺序 | `moss_lora/annotations.py:1062-1076` | `prefix + deduped tags + prose`，用 `", "` 连接 |
| finalize 四变体 | `moss_lora/annotations.py:1094-1124` | `original/long/medium/short`；tag 上限 40/40/25/12 |
| **short 带 prose** | `moss_lora/annotations.py:1123-1124` | `short = compose_caption(prefix, tags, caption.short, max_tags=12)` |
| finalize 调用 | `scripts/finalize_dataset.py:639-647` | v4 路径：`build_caption_variants(..., annotation.caption, ...)`；无 VLM 时 fallback 仍把 `source_prose` 拼进 short（645-647） |
| `caption_original` | `scripts/finalize_dataset.py:246-251` | 优先已有 `caption_original`；否则 `prefix + tags + source_prose`，上限 40 tags |
| 前缀校验 | `scripts/finalize_dataset.py:727-731` | 每个变体必须含 trigger/artist/character，否则记 error |
| clean30 单画师前缀 | `scripts/build_clean30_dataset.py:143-147` | 只用 `row["artist"]`（主画师），`unknown_if_missing=False`；多画师列表不进入 caption |
| clean30 画师规范化 | `scripts/build_clean30_dataset.py:100-101` | `lower()` + 空格转下划线 |
| clean30 行装配 | `scripts/build_clean30_dataset.py:486-530` | current 行沿用 v4 body；expansion 行用 VLM prose 或 tag fallback |
| clean30 no-artist | `scripts/build_clean30_dataset.py:550-554` | 只写 metadata，不进 `caption` 对象 |

### 2.2 prepare_captions 与 finalize 的关键差异

| 路径 | short 变体 | 代码 |
|---|---|---|
| `scripts/prepare_captions.py` | `short = prefix + 12 tags`，**无 prose** | `scripts/prepare_captions.py:864-883`，short 在 878 |
| `scripts/finalize_dataset.py`（v4 实际路径） | `short = prefix + 12 tags + caption.short`，**有一句 prose** | `moss_lora/annotations.py:1123-1124`；`scripts/finalize_dataset.py:639-647` |

v4 走的是 finalize：`moss_lora/vlm.py:48` 定义 `PROMPT_VERSION="moss-caption-v4"`，NSFW prompt 要求
`caption.short` 非空；`scripts/finalize_dataset.py` 合并 `annotation_validated`。因此 v4 的 short **不是**
prepare_captions 文档里写的纯 tag 变体。`docs/CAPTIONS.md:129-131` 描述的是 prepare_captions 的规则，
不能直接套到 v4。`[代码 + 实测]`

### 2.3 缓存侧

| 事实 | 代码/数据 |
|---|---|
| 默认变体权重 | `moss_lora/cache.py:60-67`：original .16、long/medium/short 各 .64/3、layout .20 |
| 缺失 kind 归一化 | `moss_lora/cache.py:78-101`：无 layout 的 item 把 .20 分给其余变体 |
| v2 payload 变体提取 | `moss_lora/cache.py:307-332`；`_caption_records` 在 209-305 |
| **`caption_no_artist` 不缓存** | `moss_lora/cache.py:59` 的 `CAPTION_KEYS` 不含它；提取逻辑只读 `caption_original`/`caption`/`text`/`prompt`/`image_description`/`layout` |
| 训练采样 | `moss_lora/dataset.py:51-63`：`random.choices(weights=...)`，per-item weight |
| 配置覆盖 | `moss_lora/config.py:49-73`；5ep 配置没有 `data.variant_weights`，所以用默认权重 |
| 文本编码 | `moss_lora/text.py:29-38`：chat template + tokenizer，右截断；artist 在位置 ~10，截断切不到 |

`docs/CAPTIONS.md:233` 也明确写：`caption_no_artist` 只写 metadata，当前 cache 不缓存它。
`docs/CAPTIONS.md:139-140` 说明它是为未来 artist-dropout/CFG 预留的。**5ep 运行没有把它变成缓存变体，
也没有训练期 dropout。** `[代码]`

---

## 3. 真实数据统计

### 3.1 数据集是 255 画师混合集，不是单画师

来自 `clean30_top256/report.json` 与 `metadata.jsonl`：

| 指标 | 数值 |
|---|---:|
| 总行数 | **20,953** |
| 有画师条件的真实画师 | **255** |
| 无画师行（`artist=unknown`，`artists=[]`） | **771（3.68%）** |
| current / expansion | **4,825 / 16,128** |
| 多画师 caption | **0** |
| 有 layout 的行 | **15,114（72.1%）** |
| 无 layout 的行 | **5,839（27.9%）** |

Top 20（`metadata.jsonl` 实测）：

| artist | rows | artist | rows |
|---|---:|---|---:|
| unknown | 771 | gweda | 173 |
| mignon | 298 | houkisei | 163 |
| sincos | 232 | freng | 161 |
| omone_hokoma_agm | 225 | piukute062 | 159 |
| dishwasher1910 | 224 | nekoda_(maoda) | 158 |
| mikozin | 201 | kani_biimu | 156 |
| deyui | 191 | xipa | 156 |
| niliu_chahui | 187 | haku89 | 153 |
| nekoya_(liu) | 183 | kantoku | 151 |
| esearu | 178 | unfairr | 149 |

画师长尾：min **28**，p10 **32**，p25 **40**，median **74**，p75 **101**，p90 **136**，max **298**；
`<50` 张的画师 **82** 个，`<100` 张 **190** 个，`>=200` 张只有 **5** 个。
设计文档 `style_artist_strategy.md:56` 指出**只有 19 个画师 ≥150 张**，因此 per-artist LoRA 只能作为
诊断/高端选项，不是 255 画师主产物。

### 3.2 每个画师的训练曝光：核心数字

按 `docs/bs8_5ep_launch.md:13,18,21`：20,953 items、batch 8、2,614 step/epoch、10,456 step（4 epoch）
= 83,648 个样本。**注意**：`74/20,953 = 0.353%` 是**样本占比**；一个 batch 有 8 个样本，所以
“含该画师至少 1 张图的 batch 数”约为 `1-(1-p)^8`，不是 `图片数/8`。一个画师每个 epoch 的图片通常
落在**不同的 batch** 里（每 epoch 约 74 个不同 batch），4 epoch 约 292 个 step；不是 37 个。

| 角色 | 图片数 | 4 epoch 样本数 | 含该画师至少 1 张图的 batch 数（≈4 epoch） | 占 10,456 step | 样本占比 |
|---|---:|---:|---:|---:|---:|
| 全局 trigger `@CH3OCH3` | 20,953 | 83,648 | **10,456** | **100%** | 100% |
| mignon（最大） | 298 | 1,192 | **≈1,127** | 10.8% | 1.42% |
| sincos | 232 | 928 | ≈888 | 8.5% | 1.11% |
| 中位画师 | 74 | 296 | **≈292** | **2.8%** | **0.353%** |
| p10 画师 | 32 | 128 | ≈127 | 1.2% | 0.153% |
| 尾部画师 | 28 | 112 | **≈111** | **1.07%** | 0.134% |
| unknown（无 artist tag） | 771 | 3,084 | ≈2,671 | 25.5% | 3.68% |

> `unknown` 行没有 artist tag，上表最后一行只表示“含 unknown 图片的 batch”，不是 artist 条件。
> batch 数用 `2,614 × (1 - exp(-n/2,614)) × 4` 估算；同一 batch 内重复同一画师的情况很少，
> 因此接近“每 epoch 图片数 × 4”。

**这是本报告最强的单一数字**：中位画师在整个运行中只出现在约 **292 个 optimizer step**（2.8%）的
batch 里；尾部 28 张只有约 **111 个**（1.07%）；最大画师 mignon 也只有约 **1,127 个**（10.8%）。
相比之下 trigger 出现在每一个 step（100%）。模型仍然有充分理由把全局风格绑定到 trigger，而不是
为 255 个 artist tag 分别建立方向；但“稀有”应表述为**约 1–11% 的 step**，而不是 0.13–1.4%。`[实测 + 代码]`

### 3.3 每个变体都含 artist tag，位置和占比

v4 finalize 1,000 行实测（cache 同款 chat template + tokenizer；artist span 为 `", artist:<id>"`
的 token 数）：

| 变体 | 含 artist 的行 | artist token 中位/最小/p90/max | 文本 token p50 | artist token 占比（中位/均值） |
|---|---:|---:|---:|---:|
| original | 100% | 5 / 3 / 8 / 17 | 144（cache index 154） | 3.68% / 4.03% |
| long | 100% | 5 / 3 / 8 / 17 | 292（cache 317） | 2.03% / 2.48% |
| medium | 100% | 5 / 3 / 8 / 17 | 166（cache 176） | 3.45% / 3.94% |
| short | 100% | 5 / 3 / 8 / 17 | 77（cache 87） | 6.82% / 7.66% |
| layout | 100%（真实画师行） | 5 / 3 / 8 / 17 | 272（cache 282） | 1.92% / 2.04% |

- artist tag 的起始位置：非 layout 变体 **token 10**（0-based，p50=max=10）；layout JSON 里 **token 23**
  （p50=23，max=30），因为 `{"summary":{...,"caption":"@CH3OCH3, artist:...` 前面有 JSON key。
- 没有任何真实画师行的变体缺 artist；`clean30_top256` 20,953 行检查 `artist:<id>` 子串，缺失数 **0**
  （`caption_no_artist` 除外，它本来就该缺）。`[实测]`
- 但 `scripts/build_clean30_dataset.py:143-147` 的 `trigger_parts` **只读单数 `row['artist']`**，不读
  `artists` 列表；本数据集中每个被选中的真实画师行都有非空 `artist`（缺失 0），所以没有额外丢 tag。
  若未来只通过 `artists[0]` 选中而单数字段为空，会静默丢 tag；这是必须保持的 schema 约束。`[代码 + 实测]`
- 按训练直方图加权，artist token 约占条件文本的 **4.2%**；prose 占绝对多数。

### 3.4 v4 的 tag-only 变体消失了多少？

| 数据 | `short` 有 prose | `medium` 有 prose | `long` 有 prose | `original` 有 prose | `layout.summary` 有 prose |
|---|---:|---:|---:|---:|---:|
| `clean30_top256` v1（20,953 行） | **670（3.2%）** | 5,495（26.2%） | 5,495（26.2%） | 4,825（23.0%） | 2,334/15,114（15.4%） |
| v4 finalize pilot（1,000 行） | **751（75.1%）** | 751（75.1%） | 751（75.1%） | 142（14.2%） | **810/810（100%）** |

v1 的 5,495 条 prose = 4,825 条 current 行自带的 source prose + 670 条 expansion 行的 VLM prose；
v1 的 `short` 基本是纯 tag（96.8% tag-only），这正是 prepare_captions 风格的“短 tag 变体”。
v4 finalize pilot 的 751 条 prose = 727 条 valid/repaired VLM + 24 条 source prose；
**short 的 prose 覆盖率从 3.2% 跳到 75.1%**，`long/medium/short` 三个变体同时变成 prose 变体。
`[实测]`

**完整 v4 的限制**：short 的 prose 覆盖率取决于 VLM valid rate。pilot 是 75.1%（273 条 fallback）；
如果完整 v4 的 valid rate 更高，short prose 会接近 100%；如果与 pilot 类似，则约 75%。因此 §3.5 的
“约 86% step 带 prose”是**上界估计**，保守下界约 **73%**。不要把 pilot 的 75.1% 直接当成完整 v4 的
精确值。`[推断]`

v4 pilot 的 cache index 进一步确认：

- `short` 1,000/1,000 个 item 都有（p50 87 tokens）；
- `long` 只有 727/1,000（273 个 fallback 行 `original == long`，被 cache 去重，`DATASET_FORMAT.md:226`
  的规则）；
- `medium` 997/1,000；`original` 1,000/1,000；`layout` 810/1,000；
- variant 组合：5 变体 605、无 long 的 4 变体 270、无 layout 的 4 变体 122、3 变体 3。

对完整 v4，`long` 覆盖率由直方图反推约 **98%**（§1.1），因此完整 v4 的 short/medium/long 基本每个
item 都有；约 2/3 item 额外有 layout。`[推断，强]`

### 3.5 训练 step 的 caption 形式构成

把 `docs/bs8_5ep_status.md:84` 的观测直方图（13,120 样本）与最终结构结合：

| 变体 | 观测占比 | v4 是否有 prose | 贡献的“带 prose step” |
|---|---:|---|---:|
| long | 22.7% | 是（约 98% item） | 22.7% |
| medium | 23.2% | 是 | 23.2% |
| short | 23.1% | 是（finalize 带一句） | 23.1% |
| layout | 13.2% | summary 100% 是 | 13.2% |
| original | 17.7% | expansion（77%）tag-only；current（23%）有 source prose | 17.7% × 23% ≈ **4.1%** |
| **合计带 prose** | | | **≈ 86.3%** |
| **合计 tag-only** | | | **≈ 13.6%**（且是 40-tag 长 tag 行，不是裸 prompt） |

这是 `[推断]`，但依据是：观测直方图（实测）、finalize `short` 带 prose（代码）、layout summary 在
v4 pilot 100% 带 prose（实测）、expansion/current 比例 77/23（实测）。即使把 layout summary 当成
tag-only，带 prose 也仍有 ~73%；结论不变：**纯 tag 条件在训练中最多约 1/7 的 step，而且从不是
`trigger, artist, 12 tags` 这种推理时最常用的短 prompt。**

### 3.6 风格词是通用模板，不是画师判别信号

来自 `style_artist_strategy.md:305-324` 的本地 14,063 行扫描：

| prose 词 | 命中行数 | 占扫描行 | 命中 top-5 画师占比 |
|---|---:|---:|---:|
| digital | 6,188 | 44.0% | 6.9% |
| line art | 5,266 | 39.6% | 6.4% |
| cel shading | 4,724 | 33.6% | 6.8% |
| soft shading | 3,709 | 26.4% | 6.8% |
| pastel | 2,251 | 16.0% | 7.1% |
| painterly | 1,345 | 9.6% | 7.8% |
| 3d/cgi | 1,044 | 7.4% | 7.5% |
| monochrome | 488 | 3.5% | 8.2% |
| realistic | 385 | 2.7% | 7.3% |
| watercolor | 339 | 2.4% | 8.6% |
| sketch | 260 | 1.9% | 8.8% |

top-5 current 画师本身只占 ~4.5% 图片，所以 6–9% 只是 **1.3–2 倍富集**；没有单画师占某词命中的
1.8–3.8% 以上。结论：这些是 VLM 的通用画风套话，不是画师标签。

v4 pilot 的 1,000 行复核（`ab_1k/final/metadata.jsonl`）：

- `long` prose 中位 **112 词**；**66.5%** 的 long 至少含一个风格词，平均 **2.61** 个，p90 **6** 个；
- `lighting` 38.2% 行、top-5 画师占 7.3%；`color palette` 13.6% 行、top-5 占 12.5%；
  `anime-style` 10.5% 行、top-5 占 15.2%；`anime illustration` 12.2% 行、top-5 占 14.8%；
- `pastel` 3.6%、`painterly` 0.8%、`monochrome` 2.0%、`sketch` 1.1%、`watercolor` 0.1%。

同时，booru 结构化的 style tag 极稀疏：`style_artist_strategy.md:277` 给出 **3.06%** 的行有任何
core style tag；`clean30_top256` 与 v4 pilot 里 `style:` 前缀 tag 都是 **0**。设计文档建议的
“100% 行带 `style:<family>` 的粗粒度风格层”**没有实现**。`[实测]`

### 3.7 画师 tag 的格式/大小写不一致（真实风险）

| 数据 | 情况 |
|---|---|
| `clean30_top256` v1 | 256 个 exact artist 字符串（255 真实 + unknown）；case/space 重复组 **0**；`artist:unknown` **0**；全部 lowercase_underscore |
| v4 finalize pilot | 257 个 exact artist 字符串；pilot 内部 case 重复组 0，但跨数据集出现大小写不一致：**`Sheya` 5 行** vs clean30 **`sheya` 54 行**；另外 `artist_12784` 在 pilot 有 8 行、clean30 有 **98 行**，两边都未按设计文档恢复成 `fujisaki_hikari`；并且有 **11 行 `artist:unknown`** |

原因：

- `scripts/finalize_dataset.py` 的 `_clean_prefix_value` 只做 NFKC/空白清理，**不 lower、不合并别名**
  （`moss_lora/annotations.py:88-92, 1030-1036`）；`dedupe_names` 只在**单条 caption 内**做
  case-insensitive 去重，保留第一次出现的拼写（`moss_lora/annotations.py:101-113`）。
- clean30 的 `norm_artist` 才做 `lower()` + 空格转下划线（`scripts/build_clean30_dataset.py:100-101`）。
- v4 pilot 的 11 行 `artist:unknown` 说明 finalize 路径在 `artists` 为空/含 unknown 的边界情况下会写出
  被设计文档禁止的 catch-all tag（`style_artist_strategy.md:369-370` 明确要求不要 emit `artist:unknown`）。

后果：Qwen tokenizer 区分 `Sheya`/`sheya`（大写 S vs 小写 s 是不同 token），LoRA 会把同一个画师学成
两个互不相关的 key；本来 median 只有 74 张，再对半分会把已经稀有的信号进一步稀释。推理时用户必须
使用训练时完全一致的拼写，换大小写/别名就可能完全不触发。`[实测 + 代码；完整 v4 是否中招未验证]`

**这条对完整 v4 是否成立，取决于 v4 metadata 的 base**：如果是从 `clean30_top256`（已 lowercase）重建，
则应为 0 重复；如果从 pilot/expansion 的 raw artist 字段重建，则存在分裂。远程恢复后必须审计（§7）。

### 3.8 训练充分性

| 项目 | 数值 | 来源 |
|---|---|---|
| 数据量 | 20,953 items | `docs/bs8_5ep_launch.md:13` |
| batch / step per epoch / max steps | 8 / 2,614 / 13,070 | `docs/bs8_5ep_launch.md:18,21` |
| 实际导出 step | **10,456**（epoch 4 结束） | `docs/bs8_5ep_export_and_release.md:3-7` |
| 完成度 | **80.2%**（4/5 epoch） | 同上 |
| LoRA | rank 64 / alpha 64；q,k,v,o,ffn_up_gate,ffn_down；`text_projection=false` | `configs/train_5090_v4_5ep_r64_bs8.yaml:48-51` |
| 可训练参数 | adapter 文件 594,592,688 B（fp32）≈ **149M** | `docs/bs8_5ep_export_and_release.md:28` |
| LR | 1.5e-4 cosine，warmup 200；step 10,480 时 1.45e-5 | `config:28`；`docs/bs8_5ep_pause.md:43` |
| 早期 loss | step 1,720：mean 0.4076，趋势 t=-0.14（flat） | `docs/bs8_5ep_status.md` |
| 最后记录 loss | step 10,480：0.5299（sigma 0.5673；不能与早期均值直接比） | `docs/bs8_5ep_pause.md:43-47` |
| 验证/CFG | 30 steps、CFG 3.0、negative=`nsfw, worst quality, bad anatomy, watermark, jpeg artifact` | `config:59-61`；`moss_lora/trainer.py:328-353` |

**判断**：按 `docs/FAQ.md:53-73` 的经验值（500+ 图 2–5k step；rank 64 适合大数据；lr 5e-5–2e-4），
10,456 step / 1.5e-4 / rank64 的**总 schedule 不算短**；cosine 尾部 lr 已到 1.45e-5，运行接近计划终点
（80.2%）。所以问题不是“总步数不够”，而是 **255 路把总步数摊薄**：中位画师约 292 个 batch（2.8%，见 H1）。
loss 是 flow-matching 噪声预测损失，受 sigma 和 caption 变体影响很大，**不能**用“loss 平”证明风格
没学到，也不能用它证明学到了；本次没有 held-out 风格评测。因此本报告**不把“总训练不足”列为高置信
原因**；若 E1 单画师控制仍失败，再考虑 step/lr。`[实测 + 推断]`

---

## 4. 根因排序（按证据强度）

**排序口径**：H1 是 255 路分摊造成的 artist 稀有；H2 的 prose 捷径是在此之上放大。两者不是互斥病因。
按可干预性排序，**先修 255 路绑定/频率（top128、artist-balanced sampling），再修 prose 捷径（tag-only +
去风格 prose）**；“finalize 的 short 带 prose / 短 tag 变体丢失”是**次级 confound 和加速器**，不是唯一
主因。下面的 H1/H2 并列第一梯队，H3–H5 依次为条件/设计/训练因素。

### H1（第一梯队）：255 路 artist 绑定 + 稀有样本 + 共享 trigger

**机制**：
- 数据集是 255 个真实画师的混合集：82 个 <50 张、190 个 <100 张，中位 74 张，尾部 28 张。
- 一个 rank64/149M 的 adapter 要同时编码 255 个风格方向 + 角色/内容；每个画师约 111–1,127 个 batch
  （1.1–10.8% 的 step）。
- artist tag 本身不弱（位置 10、5 tokens、~4.2% 条件 token），但每个 artist 只占 **0.35%** 的样本；
  中位约 **292 个 batch（2.8%）**。trigger `@CH3OCH3` 出现在 **100%** 样本，模型可以把全局风格绑定到 trigger。
- 设计文档自己在 `style_artist_strategy.md:392` 给出 fallback：**artist fidelity 弱时先退到 top 128
  （≥75 张），不要先加 rank**。

**证据强度：强（画师分布/曝光实测；token 占比实测）。** 如果 E1 单画师成功而 255 路失败，H1 成立；
E2 top128/平衡采样应提升中尾部画师。

**预期 A/B 结果**：E1 单画师控制明显好转；E2 提升 mid/tail；只加 step/rank 不会根治。

### H2（第一梯队，与 H1 并列）：prose 捷径 / 短 tag 变体丢失（放大器，不是唯一主因）

**机制**：
- v4 finalize 的 `short` 带一句 prose（`moss_lora/annotations.py:1123-1124`）；v4 pilot 中
  short/medium/long 的 prose 覆盖率都是 **75.1%**，而 clean30 v1 的 short 只有 **3.2%** prose。
  完整 v4 的 short prose 覆盖率取决于 VLM valid rate，标 `[推断]`。
- 按训练直方图与结构推导，约 **86% 的 step 带 prose**（保守下界约 73%），只有约 **14%** tag-only，
  且 tag-only 是 40 个 tag 的长行。推理时 `trigger + artist + 少量 tags` 在训练分布里几乎不存在。
- prose 的风格词是通用模板（`digital`/`cel shading`/`soft shading` 等 top-5 画师只占 6–9%），模型可以用
  prose 解释风格，不必读 artist tag。
- **分变体看，short 不是主要风格携带者**：v4 pilot 的 style-term 命中约为 long **1.82/行**、
  medium **0.56/行**、short **0.07/行**。short 的 prose 更多是内容/构图，真正提供风格捷径的是
  long/medium。因此“短 tag 变体丢失”是次级因素；主要 OOD 来自 long/medium 的 prose。
- `original` 仍有 16%（有 layout）到 20%（无 layout）的权重；expansion 行（77%）的 `original` 是
  tag-only，但它是 40 个 tag 的长行，不是推理时常用的 `trigger + artist + 少量 tags`。所以不是
  “完全没有 tag-only”，而是“没有短的、裸的 tag-only”。

**证据强度：强（v4 pilot 实测 + 代码）；完整 v4 的 prose 比例是 `[推断]`。这是次级 confound/放大器，
不是唯一主因；但它是 E3 最直接可干预的杠杆。**

### H3（强）：artist dropout 未实现，推理没有可校准的 artist 方向

- 设计文档要求 **10% artist dropout**（`style_artist_strategy.md:375`），但 cache 不缓存
  `caption_no_artist`（`moss_lora/cache.py:59`；`docs/CAPTIONS.md:233`）。
- 训练循环没有 text/artist dropout；验证 negative 是固定质量词，不是“同一 caption 去掉 artist”
  （`config:61`；`moss_lora/trainer.py:328-353`）。
- 数据里只有 771 条 unknown 行（3.68%）没有 artist tag，且它们是**不同图片**，不是同一张图的无条件
  配对，不能替代 dropout。

**证据强度：强（代码/配置）；对风格失效的贡献是“无法用 CFG 放大 artist”，不是直接原因。**

### H4（中强）：设计中的稠密 `style:` 层没有落地

- 设计文档结论：artist 是唯一稠密、可判别的风格载体，booru style tag 只覆盖 3.06%，VLM prose 通用
  （`style_artist_strategy.md:44-48, 305-324`）；建议 100% 行加 `style:<family>`。
- 实测 `clean30_top256` 与 v4 pilot 的 `style:` tag 都是 **0**。
- 结果：模型没有“粗粒度风格”这个稠密中间变量；只能要么读稀有的 artist tag，要么读通用的 prose。
  这会让 artist tag 既是唯一风格源、又太稀有，放大 H1/H2。

**证据强度：中强（设计 vs 实现的缺口是实测；对最终画质的因果需 A/B）。**

### H5（弱-中）：总 schedule 不短，但 per-artist 曝光被 255 路摊薄

- 按 `docs/FAQ.md:53-73`，10,456 step 对 20,953 图属于充足区间（500+ 图建议 2–5k step）；lr 1.5e-4
  在 5e-5–2e-4 内；cosine 尾部 1.45e-5，接近计划终点。导出 epoch 4 少 20%，但不是主因。
- 真正的“少”是 per-artist：中位约 292 个 batch、尾部约 111 个（H1），不是全局 step 不够。
- loss 早期 flat、最后 0.5299，但 loss 不是风格指标；没有 holdout 图，无法判断“条件设计错误”还是
  “per-artist 容量/曝光不足”。
- rank 64/149M 不算小，所以“加 rank”不是第一优先级；设计文档也建议先 top128。

**证据强度：弱-中。** 单独加 step/rank 不应作为首选，且无法解释为什么 artist tag 在推理时几乎无效。

### H6（中，需远程确认）：tag 大小写/别名分裂

- v4 pilot 真实存在 `Sheya` vs clean30 `sheya` 的大小写分裂；另外 `artist_12784` 在 clean30 仍有 98 行
  （pilot 8 行），设计文档要求恢复成 `fujisaki_hikari` 但未执行；v4 pilot 还有 11 行 `artist:unknown`。
- 如果完整 v4 metadata 继承了 raw case/alias，同一画师的 tag 会被拆成多个 key；中位 74 张再对半，
  直接掉到 ~37 张/epoch、约 **147 个 batch**（4 epoch），而不是 74 张的约 292 个。
- 如果完整 v4 是从 clean30 的 lowercase 行重建，则这条不成立；必须远程审计。

**证据强度：中（pilot 实测；完整 v4 未验证）。**

### H7（中，机制性）：trigger 是全局风格 token，artist 是稀有修饰符

- `@CH3OCH3` 6 个 token，出现在 100% 样本；artist 5 个 token，中位 0.35%。
- 在 causal Qwen TE 中，trigger 在 artist 之前；artist token 能看到 trigger，trigger 看不到 artist。
  DiT 侧文本 token 先经过 2 个无 causal mask 的 text block（`moss_image.py:144-155`），再与 image token
  做 joint self-attention（`moss_image.py:290-291`），不是 cross-attention。因此 artist token 的最终
  hidden state 会被整段 prose 上下文混合；推理时去掉 prose，artist 的表示发生分布偏移。
- 结果：trigger 很容易学成“整个数据集的平均风格”；artist 学成弱修饰。只给 artist 不给 trigger 时
  可能更差，给 trigger 时又回到数据集平均风格。

**证据强度：中（token 频率/结构实测；注意力行为未可视化）。**

### H8（弱/未验证）：数据本身的画师风格异质性

- 255 个画师的头像/风格是否稳定、是否有画风漂移、是否有 mislabel，本次**没有做图像判断**。
- 本地只能确认 tag 数量、画师分布、caption 结构；不能确认“mignon 的 298 张是否同一风格”。
- 需要 holdout 图和人工/embedding 聚类才能判断。

**证据强度：弱（无图像证据）。**

### 明确排除/降级

- **多画师同图稀释**：`artists` 列表长度 >1 的行 **0**；`build_clean30_dataset.py:143-147` 只取主画师。
  所以“多个 artist tag 互相稀释”不是本数据集的原因。`[实测]`
- **artist tag 被截断**：artist 在 token 10；即使 `max_text_length=512` 也切不到；实际 long p99
  ~475 tokens（v4 pilot），layout p99 ~654。`[实测]`
- **A1111 `(artist:foo:1.3)` 语法**：仓库没有权重解析，会破坏 `artist` token；不要用。
- **`caption_no_artist` 被缓存**：没有；`CAPTION_KEYS` 不含它。`[代码]`

---

## 5. 受控 A/B 实验矩阵

> 共同前提：固定 seed、固定 sampler（30 steps、CFG 3.0、同一 negative prompt）、固定分辨率
> 1024²、每个 checkpoint 用同一组 prompt；所有对比只改一个变量。评测集用**训练中未出现的 holdout
> 图片**（每个画师 10 张，按 `image_sha256` 去重；至少 2 个头部、4 个中部、2 个尾部画师）。

### E1：单画师控制（最强诊断，最便宜）

| 项目 | 内容 |
|---|---|
| 目的 | 区分“255 路干扰/稀有 tag” vs “caption/条件/pipeline 本身坏了” |
| 数据 | 只取 `mignon`（298 张，最大）或一个中位画师（74 张）；同一批 caption 规则；holdout 20 张不训练 |
| Caption/cache | 与 5ep 完全相同的 finalize 规则；重建 1M cache；保留 `short` 的 prose（不要偷偷改） |
| 训练 | 同 rank64/alpha64、同 lr 1.5e-4 cosine、同 4 epoch（或按图片数折算到相同 step 数） |
| 隔离什么 | 如果单画师 LoRA 能复现风格 → H1 成立（255 路/稀有 tag 是主因）；如果仍不能 → H3/H4/H5 或 caption 设计是主因 |
| 评测 | 8 个 prompt × 5 seed；与 10 张 holdout 的 DINO/CLIP 最近质心 top-1；人工三选一 |
| 预期 | 单画师应显著优于 255 路；若没有，优先查 caption/under-training |

### E2：top128 或 artist-balanced sampling（针对长尾）

| 项目 | 内容 |
|---|---|
| 目的 | 提高每个 artist 的样本频率，隔离 H1 |
| 数据 A（设计文档 fallback） | top128（≥75 张，15,132 行，设计文档 `style_artist_strategy.md:392` 的表）；重建 metadata/cache |
| 数据 B（更彻底） | 保持 top256，但用 inverse-frequency `WeightedRandomSampler` 或“每个 batch 尽量 8 个不同画师”的 batch sampler；cache 不变，只改 dataset/sampler |
| 训练 | 总 step 与 5ep 相同（10,456）或按 epoch 对齐；其他超参不变 |
| 隔离什么 | 如果 top128/平衡采样显著提升中尾部画师 → 稀有条件是主因；如果头部画师反而变差 → 容量/干扰 |
| 评测 | 按画师分层报告：head/mid/tail 各自 top-1 和 margin；不要只报总体平均 |
| 注意 | top128 会减少总行数，step/epoch 变化；必须固定**总 step**或**总样本数**，不能同时改 |

### E3：Caption 去混淆（tag-only 变体 + 去风格 prose，两个 arm）

| Arm | Caption/cache 改动 | 隔离什么 |
|---|---|---|
| **E3a：真 tag-only 变体** | 新增 `tag_only = @CH3OCH3, artist:<id>, character:<id...>, 12–25 general tags`（无 prose）；重建 cache；把 `tag_only` 权重设为 **≥0.5**，其余 long/medium/short/original/layout 分剩余权重 | “推理时短 tag prompt 在训练中不存在”这一分布偏移 |
| **E3b：去风格 prose** | 保留 long/medium/short 的结构和内容，但用规则/LLM 删除 style/lighting/palette/medium 形容词（如 `soft lighting`、`cel shading`、`pastel palette`、`anime-style`、`detailed`），保留人物/动作/NSFW/文字/构图；重建 cache | “prose 风格词提供捷径”这一内容因素 |
| 训练 | 同 5ep 超参、同总 step | — |
| 评测 | 同一 prompt matrix；E3a 重点看 P1（裸 tag）是否接近 P4（带 prose）；E3b 重点看 P1 是否提升、P4 是否下降 | — |
| 预期 | 若 E3a 大幅提升 P1 → 分布偏移是主因；若 E3b 提升 P1 而 P4 下降 → prose 捷径是主因；两者可叠加 | — |

**注意**：E3a 不能只调 `--variant-weights short=0.5`。v4 的 `short` 仍带一句 prose；那只是“更短一点的
prose”，不是 tag-only。必须新增真正的 tag-only 文本。

### E4：Artist dropout + CFG（把 artist 变成可控方向）

| 项目 | 内容 |
|---|---|
| 目的 | 给模型一个“同一图片、无 artist”的对照分支，让推理时能用 CFG 放大 artist；隔离 H3 |
| Cache 改动 | 让 `caption_variant_records` 读取 `caption_no_artist`，产出 `no_artist` 变体；或让 `CachedDataset` 同时返回 `text` 和 `text_no_artist`；权重 10–15% |
| 训练改动 | 按样本以 p=0.10–0.15 把 context 替换成 no-artist 版本（标准 CFG dropout）；不要只把它当普通 variant 采样（那样和 caption 长度耦合） |
| 推理 | positive = 正常 caption；negative = 同一 caption 去掉 artist（或 `artist:unknown`）；扫 CFG 1.5/3.0/5.0 |
| 隔离什么 | 如果 CFG 能明显提高 artist 忠实度 → 模型其实学到了 artist 方向，只是默认被 prose/trigger 盖住；如果 CFG 无效 → artist 根本没绑定 |
| 代价 | 多一份文本编码/缓存；p 太大会削弱 artist；需要额外训练步数 |

### 统一评测协议（所有实验共用）

**Prompt matrix（每个画师、每个 seed）**：

| 编号 | Prompt | 测什么 |
|---|---|---|
| P1 | `@CH3OCH3, artist:<id>, 1girl, solo, upper body` | 裸 tag：artist 是否能独立控制风格 |
| P2 | `artist:<id>, 1girl, solo, upper body`（无 trigger） | trigger 是否在替 artist 干活 |
| P3 | `@CH3OCH3, 1girl, solo, upper body`（无 artist） | 无 artist 基线 |
| P4 | `@CH3OCH3, artist:<id>, <一句训练集风格 prose>` | prose 是否在替 artist 干活 |
| P5 | `@CH3OCH3, artist:<id>, <仅内容 tags>` | 内容 tags 是否足够 |

**指标**：

1. **Artist 最近质心 top-1**：用 DINOv2 或 CLIP image embedding，计算每个画师 10 张 holdout 的质心；
   生成图与 own-artist 质心 vs 其他画师质心的 cosine margin；报告 top-1 和 margin。
2. **Tag 效应**：`P1 - P3`（artist 是否有效）；**Prose 效应**：`P4 - P1`（prose 是否在替代 artist）；
   **Trigger 效应**：`P1 - P2`（trigger 是否在替代 artist）。
3. **人工三选一**：每个生成图 vs 同画师/其他画师 holdout，3 名评审盲选；报告准确率和一致性。
4. **内容保真**：角色/构图 tag 的检测或人工检查，避免“风格对了但内容崩了”被误判为成功。
5. **固定 seed**：至少 3 个 seed（42/123/2026），报告均值±标准差；不要挑图。

**样本量建议**：6 个画师（2 head、2 mid、2 tail）× 5 prompt × 3 seed = 90 张/实验；holdout 10 张/画师
只做参考，不参与训练。这个量足够看出 P1 vs P3/P4 的显著差异，又不至于评测成本失控。

### 关于“缓存权重实验”的明确结论

**不要**把 `--variant-weights short=0.5,long=0.2,...` 当成“提高 artist 权重”：
- 缓存没有 per-token weight；`variant_weights` 只改变采样哪个 caption 变体；
- 所有缓存变体都含 artist tag（真实画师行 100%）；
- v4 的 `short` 仍带一句 prose，提高 short 权重只是“更多短 prose”，不是更多 artist 条件。

**唯一有意义的权重实验**是 E3a：新增真正的 tag-only 变体，并给它 ≥0.5 权重。这不是“缓存时给 artist
加权”，而是“改变训练时 caption 形式的分布”。`[代码 + 实测]`

---

## 6. 直接回答两个问题

### Q1：artist tag 强度 vs 风格 vocabulary，哪个是主因？

**artist tag 本身不弱，主因是“artist 条件太稀有 + prose 风格词太通用”。**

- artist tag：每个变体都在、位置 10、5 tokens、~4.2% 条件 token；没有被截断，也没有被 layout 挤掉。
- 但每个 artist 只占 **0.35%** 样本（中位约 **292 个 batch，2.8% 的 step**）；trigger 占 **100%**。
- 约 **86%** 的 step 带 prose；prose 的风格词在 14,063 行扫描中覆盖 90.9%，但 top-5 画师只占 6–9%
  （base 4.5%），是通用套话，不是画师判别信号。
- 因此模型的最优策略是：**用 trigger + 通用风格词拟合训练集，把 artist tag 当弱修饰**。推理时没有
  prose，artist tag 又稀有，风格自然不匹配。

一句话：**不是“artist tag 太弱”，是“artist tag 太稀有，而 prose 太容易解释风格”。**

### Q2：应该在缓存时提高 artist 权重吗？

**不应该，也不能。**

- 缓存里没有“artist 权重”这个字段；`variant_weights` 只选择 caption 变体，不改变 artist 的有无/强度。
- 直接缩放 artist span 的 hidden state 会被 RMSNorm 基本抵消，且训练/推理不可复现，ComfyUI 也带不走。
- 重复 artist tag 只是增加几个 token，不是乘性权重，还会改变分布、占 token 预算。
- A1111 `(artist:foo:1.3)` 不被解析，反而会把 `artist` 拆成不同 token，可能触发 `artist:unknown`。

**正确顺序**：
1. **先做推理-only prompt matrix**（P1–P5）在现有 checkpoint 上确认 artist 是否有效；这一步不训练，
   成本最低。
2. **Caption 去混淆**：新增真 tag-only 变体（≥50% 权重）+ 去掉 prose 里的通用风格形容词（E3）。
3. **Artist dropout 10–15% + CFG negative = 同 caption 去掉 artist**（E4）。
4. **top128 或 artist-balanced sampling**（E2）；如果画师忠实度是产品核心，19 个 ≥150 张的画师做
   per-artist LoRA，而不是一个 255 路 adapter。
5. 最后才考虑 rank/step 扩展；设计文档也建议先 top128 再 rank。

---

## 7. 远程数据审计命令（SSH 恢复后执行）

> 目的：确认完整 v4 metadata 的变体覆盖、artist tag 拼写、`max_text_length`、实际 variant 直方图，
> 以及 cache index 与 metadata 的对应关系。以下命令只读。

### 7.1 找到 v4 metadata 与 cache

```bash
ssh user@<other-host> '
  find /home/user -maxdepth 4 -name metadata.jsonl -size +50M -printf "%s %p\n" 2>/dev/null
  ls -la /home/user/cache/clean30_v4_1m/index.json
  du -sh /home/user/cache/clean30_v4_1m
'
```

### 7.2 Cache index：variant 覆盖、权重、token 长度

```bash
ssh user@<other-host> 'python3 - <<PY
import json, collections, statistics
p="/home/user/cache/clean30_v4_1m/index.json"
idx=json.load(open(p))
items=idx["items"]
print("items",len(items),"max_text_length",idx.get("max_text_length"),"overrides",idx.get("variant_weight_overrides"))
print("sets",collections.Counter(tuple(i["variant_kinds"]) for i in items).most_common(12))
print("kinds",collections.Counter(k for i in items for k in i["variant_kinds"]))
L=collections.defaultdict(list)
for i in items:
    for k,n in zip(i["variant_kinds"],i["text_lengths"]): L[k].append(n)
for k,v in L.items():
    v=sorted(v); print(k,len(v),"p50",v[len(v)//2],"p90",v[int(len(v)*.9)],"p99",v[int(len(v)*.99)],"max",v[-1])
PY'
```

**判读**：
- `sets` 中 5 变体（original/long/medium/short/layout）占比应约 66%；无 layout 的 4 变体约 34%。
- `long` 的 count 应接近 20,953（≥98%）；如果只有 ~2,600，说明缓存不是 v4 recaption，或 long 被
  `original == long` 大量去重。
- `max_text_length` 记录实际截断上限（512/1024/4096）；artist 在 token 10，任何值都安全。

### 7.3 Metadata：画师拼写、unknown、变体内容

```bash
ssh user@<other-host> 'python3 - <<PY
import json, re, collections
p="/home/user/clean30_top256_v4/metadata.jsonl"   # 用 7.1 找到的实际路径替换
rows=[json.loads(l) for l in open(p)]
print("rows",len(rows))
c=collections.Counter(r.get("artist") or "unknown" for r in rows)
print("artists_exact",len(c),"unknown_rows",c.get("unknown",0))
groups=collections.defaultdict(list)
for a in c: groups[re.sub(r"\\s+","_",a.strip().lower())].append(a)
dup={k:v for k,v in groups.items() if len(v)>1}
print("case_space_dup_groups",len(dup),"rows",sum(c[a] for v in dup.values() for a in v))
print("top20",c.most_common(20))
for field,nt in (("caption_original",40),("long",40),("medium",25),("short",12)):
    miss=prose=0; n=0
    for r in rows:
        t=r.get("caption_original") if field=="caption_original" else (r.get("caption") or {}).get(field)
        if not isinstance(t,str) or not t: continue
        n+=1
        a=r.get("artist") or ""
        if a and a!="unknown" and f"artist:{a}" not in t: miss+=1
        # 粗略 prose 判定：去掉 prefix 和 tags 后仍有字母内容
        s=t
        for tag in ([f"@CH3OCH3"]+([f"artist:{a}"] if a and a!="unknown" else [])+[f"character:{x}" for x in (r.get("character") or [])[:2]]+list((r.get("tags") or [])[:nt])):
            if s.startswith(tag): s=s[len(tag):].lstrip(" ,")
        if re.search(r"[A-Za-z]{3}",s): prose+=1
    print(field,"rows",n,"missing_artist",miss,"prose_like",prose)
print("artist_unknown_tags",sum(1 for r in rows if "artist:unknown" in json.dumps(r.get("caption") or {})))
print("style_prefix_tags",sum(1 for r in rows for t in (r.get("tags") or []) if str(t).startswith("style:")))
PY'
```

**判读**：
- `case_space_dup_groups` 必须为 **0**；如果 >0，v4 的 artist tag 被大小写/空格/别名拆开，先修数据再谈
  artist 强度。
- `missing_artist` 必须为 0（`caption_no_artist` 除外）。
- `artist_unknown_tags` 应为 0；若 >0，说明 `artist:unknown` catch-all 进入了训练。
- `style_prefix_tags` 应为 0（当前实现没有 style 层）；若 >0，说明设计文档的 style 层已落地，需重新评估
  H4。

### 7.4 用 caption hash 反查 metadata 与 cache 是否对应

```bash
ssh user@<other-host> 'python3 - <<PY
import json, hashlib
idx=json.load(open("/home/user/cache/clean30_v4_1m/index.json"))
print("index items",len(idx["items"]))
print("first item keys",list(idx["items"][0]))
print("first caption_sha256",idx["items"][0]["caption_sha256"][:2])
print("first text_lengths",idx["items"][0]["text_lengths"])
PY'
```

把 metadata 的每个变体做 `sha256(text.encode())`，与 index 的 `caption_sha256` 对齐；若对不上，说明
metadata 不是该 cache 的源文件（例如缓存是更早/更晚的 caption 版本）。这是判断“本地 v1 metadata 是否
就是 5ep 缓存来源”的唯一硬证据。

### 7.5 训练日志与指标

```bash
ssh user@<other-host> '
  tail -5 /home/user/runs/clean30_v4_5ep_r64_bs8.log
  tail -5 /home/user/runs/clean30_v4_5ep_r64_bs8/metrics.csv
  python3 - <<PY
import csv
rows=list(csv.DictReader(open("/home/user/runs/clean30_v4_5ep_r64_bs8/metrics.csv")))
print("rows",len(rows),"last",rows[-1] if rows else None)
for k in ("loss","grad_norm","lr","sigma"):
    v=[float(r[k]) for r in rows if r.get(k) not in (None,"")]
    if v: print(k,"first",v[0],"last",v[-1],"min",min(v),"max",max(v))
PY'
```

---

## 8. 主要文件:行索引

| 主题 | 引用 |
|---|---|
| 前缀/变体构造 | `moss_lora/annotations.py:1035-1060`, `:1062-1076`, `:1094-1124` |
| short 带 prose | `moss_lora/annotations.py:1123-1124` |
| finalize 调用/ fallback | `scripts/finalize_dataset.py:639-647`, `:246-251`, `:727-731` |
| prepare short 无 prose | `scripts/prepare_captions.py:864-883`（short 在 878） |
| clean30 画师规范化/前缀 | `scripts/build_clean30_dataset.py:100-101`, `:143-147` |
| clean30 行装配/no-artist | `scripts/build_clean30_dataset.py:486-530`, `:550-554`, `:561-570` |
| cache 权重/归一化 | `moss_lora/cache.py:60-67`, `:78-101` |
| `caption_no_artist` 不缓存 | `moss_lora/cache.py:59`, `:209-305`, `docs/CAPTIONS.md:233` |
| 变体提取/采样 | `moss_lora/cache.py:307-332`; `moss_lora/dataset.py:51-63` |
| 配置覆盖 | `moss_lora/config.py:49-73` |
| 文本编码 | `moss_lora/text.py:29-38` |
| DiT 文本路径 | `moss_lora/model/moss_image.py:144-155`, `:283`, `:290-291`, `:299` |
| 验证 negative / 无 dropout | `moss_lora/trainer.py:328-353`; `configs/train_5090_v4_5ep_r64_bs8.yaml:59-61` |
| 训练配置 | `configs/train_5090_v4_5ep_r64_bs8.yaml:5,20,26,28,48-51` |
| 观测 variant 直方图 | `docs/bs8_5ep_status.md:84` |
| 最后 loss | `docs/bs8_5ep_pause.md:43-47` |
| 导出 epoch/完成度 | `docs/bs8_5ep_export_and_release.md:3-7` |
| 数据量/step | `docs/bs8_5ep_launch.md:13,18,21` |
| 设计文档：artist 唯一稠密风格载体 | `/mnt/data/train_test/style_artist_strategy.md:44-48` |
| 设计文档：prose 通用、top-5 6–9% | `style_artist_strategy.md:305-324` |
| 设计文档：10% artist dropout | `style_artist_strategy.md:375` |
| 设计文档：top128 fallback | `style_artist_strategy.md:392` |
| 数据集 schema/权重/行数 | `/mnt/data/train_test/clean30_top256/DATASET_FORMAT.md:95-101,198-205,536-567` |
| v4 pilot 报告 | `/mnt/data/train_test/ab_1k/final/report.json`, `report.md` |
| v4 pilot cache index | `/mnt/data/train_test/ab_1k/cache_1m/index.json` |

---

## 9. 执行摘要（5 条）

1. **这是 255 画师混合集，不是单画师**：20,953 行 = 255 个有画师条件的画师（20,182 行）+ 771 条无画师行（3.68%）；画师中位 74 张、尾部 28 张、头部 mignon 298 张；只有 19 个画师 ≥150 张。
2. **artist tag 不弱，但样本频率太低**：每个真实画师行的缓存变体都含 `artist:<id>`，位置约 token 10，中位 5 tokens（占条件文本 2.0–7.7%，加权约 4.2%）；但中位画师只出现在 **约 292/10,456 个 step（2.8%）**，样本占比 0.353%，而全局 trigger `@CH3OCH3` 出现在 **100%** 样本。
3. **v4 的短 tag 变体被 prose 化（次级 confound）**：clean30 v1 的 `short` 96.8% 是 tag-only（prose 仅 3.2%）；v4 finalize 的 `short` 在 pilot 75.1% 行带一句 prose（`moss_lora/annotations.py:1123-1124`），约 **86% 的训练 step 带 prose**（保守下界约 73%；完整 v4 的 valid rate 未验证），纯 tag-only 只剩约 14%（且是 40-tag 长行）。它不是唯一主因，但会让推理时 `trigger + artist + 少量 tags` 变成 OOD。
4. **prose 风格词是通用模板**：14,063 行扫描中 `digital` 44.0%、`cel shading` 33.6%、`soft shading` 26.4%、`pastel` 16.0%；这些词的 top-5 画师占比只有 6–9%（top-5 画师本身约占 4.5%）。模型用“trigger + 通用风格词”就能解释训练图，不需要读 artist tag。
5. **不要调“缓存 artist 权重”**：缓存没有 per-token weight，`variant_weights` 只选 caption 变体，所有变体都含 artist。正确顺序：先跑 P1–P5 prompt matrix + E1 单画师控制；确认是 255 路问题就先 **top128 / artist-balanced sampling**；再上真 tag-only 变体（≥50% 权重）+ 去风格 prose；再做 10–15% artist dropout + CFG；19 个 ≥150 张的画师可做 per-artist LoRA。不要缩放 hidden state、重复 tag、或用 A1111 `(artist:foo:1.3)`。
