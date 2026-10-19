# 混合画师 Caption 预处理

`scripts/prepare_captions.py` 把标注/采样得到的原始 prompt 转换成 MOSS-Image LoRA 训练用的
`metadata.jsonl`。它解决四件事：

1. **保留画师条件**：long/medium/short 三个变体都带 `artist:<id>`，支持多画师混合训练；
2. **只保留一种 prose**：中文或英文二选一，避免同一张图同时监督两种语言的描述；
3. **清理噪声**：去掉 `@handle` 水印、`#vlm` 标记和常见 quality negative；
4. **量化风险**：报告 token 长度、文字/手部覆盖、重复样本和分桶中心裁剪比例。

输出目录可以直接作为 `moss-lora-cache --input` 使用：

```text
<output>/
├── images/000001.jpg          # 指向源图的相对 symlink（或 --copy-images 的副本）
├── metadata.jsonl             # 每行一个训练样本
├── report.json                # 机器可读统计
└── report.md                  # 人类可读统计
```

源目录只读；脚本不会修改、移动或删除源图片与 sidecar。

---

## 1. 输入格式

### 1.1 原始 `.txt` sidecar

sidecar 与图片同名。典型结构是「tag 行 + 中文段落 + 英文段落 + `#vlm`」：

```text
@example_handle, artist:alice, character:hero, landscape, sunset, city lights, sign

这是一段中文描述，画面中有招牌文字。

An English description with a sign.

#vlm
```

解析规则：

- **tag 行**：第一条非空行，且以 `@` / `＠` 开头，或包含 `artist:` / `character:`；
- **`@handle`**：视为来源/水印，写入 `source_handle`，**不进入 caption**；多个 handle 只保留第一个；
- **`artist:` / `character:`**：分别提取；character 可以重复出现，按规范化名称去重；
- **剩余逗号项**：positive tags，保持原始顺序；
- **`#vlm`**：任意位置移除，不进入 caption；
- **其余非空行**：prose。按 CJK 比例分到中文块或英文块；同一语言的所有段落合并为 long prose。

如果 sidecar 只有 tag 行（没有 prose），仍然可以生成 caption；报告会把它计入
`language.fallback_to_no_prose`。

### 1.2 JSON sidecar

支持 `.json`，也支持内容以 `{` 开头的 `.txt`（按内容而不是后缀解析）。字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `artist` | string / list | 画师；有多个时取第一个，并参与 artist 计数 |
| `character` | string / list | 角色；去重 |
| `extra_tags` | list / string | 额外 tag；字符串按逗号切分；`tags` 是别名 |
| `image_description` | string | prose；若内容本身是原始 prompt（以 `@` 或 `artist:` 开头）则递归解析 |
| `caption` | string / object | 字符串按原始 prompt 或 prose 解析；`{long, medium, short}` 取第一个非空项作为 prose；layout dict 作为 layout 透传 |
| `preference_level` | any | 原样写入 metadata |
| `layout` | object | 原样写入 metadata；若没有 prose，则从 `summary.caption` / item 文本回退 |
| `source_handle` | string | 可选；显式来源 handle（保留 `@`），不进入 caption |

`caption` 的优先级高于 `image_description`，与 `moss-lora-cache` 的字段优先级一致。

### 1.3 `--input` 为 JSONL

每行一个 JSON 对象，图片路径用 `file_name` 或 `image`，相对 JSONL 所在目录解析。行内可以直接放
`artist` / `character` / `extra_tags` / `image_description` / `caption` / `layout`；没有 caption 字段时
回退到图片同名 sidecar。相同图片的重复行只处理第一行，并在
`sidecars.duplicate_manifest_rows` 中计数。

### 1.4 目录内的 `metadata.jsonl`

目录模式会优先读取根目录的 `metadata.jsonl`，行内字段与 JSON sidecar 相同；没有
`metadata.jsonl` 时使用同名 `.json` / `.txt`。两者同时存在时 manifest 行为准。
如果 manifest 行只有 `file_name` / `preference_level` 而 caption 来自同名 sidecar，
`preference_level` 会合并进最终 metadata。空的 `.json` 会回退到非空 `.txt`。

---

## 2. 规范化与清理

### 2.1 名称

- Unicode NFKC；全角 `＠` / `：` / `，` 转半角；
- 连续空白折叠为一个空格；
- artist/character 去重不区分大小写、下划线与空格（`hero` 与 `hero` 等价，
  `hero_(alt)` 与 `hero (alt)` 视为同一角色）；
- `--artist-map map.json` 把原始画师名映射到统一 id（key 匹配同样忽略大小写/下划线）：

```json
{"old_name": "canonical_name", "alias": "canonical_name"}
```

映射为空字符串时落到 `unknown`。报告中的 artist 计数使用映射后的名称。

### 2.2 Negative tag

以下 tag 会从 positive tags 中删除，不进入任何 caption 变体：

```text
worst quality, bad anatomy, watermark, jpeg artifacts,
bad hands, extra fingers, missing fingers
```

匹配忽略大小写、下划线和空格（`worst_quality` 等价于 `worst quality`）。只清理 tag；prose 原文保留，
避免误删“画面中没有水印”这类描述。

### 2.3 画师缺失

没有 `artist:` 的图片会以 `artist:unknown` 写入 caption 和 metadata，并在
`report.artists.missing_artist_items` 中计数。`unknown` **不**计入唯一画师数、Top 20 和长尾分布；
这样画师统计只反映真实条件。所有变体仍然满足 `trigger, artist:<id>, character:<ids>` 的结构，
同时明确标记条件缺失。

---

## 3. Caption 变体

每条记录包含三个变体：

```text
long   = <trigger>, artist:<id>, character:<id...>, <最多 max-tags 个 tag>, <完整 prose>
medium = <trigger>, artist:<id>, character:<id...>, <最多 25 个 tag>, <第一段 prose>
short  = <trigger>, artist:<id>, character:<id...>, <最多 12 个 tag>
```

- tag 顺序沿用源 tag 行（tag 风格通常已按重要性排序），截断时取前 N 个；
- `--max-tags` 默认 40，同时限制 long；medium/short 的上限分别是 25/12；
- `--language en|zh|auto`：`en` 只用英文 prose，`zh` 只用中文 prose，`auto` 在存在中文块时选中文，
  否则英文。**任何时候都不会同时包含两种 prose**；请求的语言没有对应 prose 时该样本没有 prose，
  但 tag/artist/character 仍然保留；
- `caption_no_artist`：与 long 相同的 tags + prose，但**不带** `artist:<id>`，用于未来做
  artist-dropout / CFG 实验。它仍保留 `character:`。

`metadata.jsonl` 行示例（SFW 合成数据）：

```json
{"file_name":"images/000001.jpg","caption":{"long":"mystyle, artist:alice, character:hero, landscape, sunset, city lights, A city street at dusk.","medium":"mystyle, artist:alice, character:hero, landscape, sunset, city lights, A city street at dusk.","short":"mystyle, artist:alice, character:hero, landscape, sunset, city lights"},"artist":"alice","character":["hero"],"tags":["landscape","sunset","city lights"],"source_handle":"@example","caption_no_artist":"mystyle, character:hero, landscape, sunset, city lights, A city street at dusk."}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `file_name` | 相对输出目录，例如 `images/000001.jpg` |
| `caption` | `{long, medium, short}`，直接交给 cache |
| `artist` | 规范化/映射后的画师 id；缺失为 `unknown` |
| `character` | 角色 id 列表 |
| `tags` | 清理后的全部 positive tags（不受 `--max-tags` 截断） |
| `source_handle` | 原始 `@handle`（保留 `@`），不进入 caption |
| `caption_no_artist` | 无画师条件的 long 变体 |
| `preference_level` | 输入提供时原样写入；未提供时为 `null` |
| `layout` | 仅当输入提供时写入 |

`moss-lora-cache` 读取 `caption` object 的三个非空变体；如果 long 与 medium 完全相同（例如只有一个
英文段落且 tag 不超过 25 个），cache 会自动去重，因此实际缓存的变体数可能是 2 而不是 3。

---

## 4. 过滤与画师不平衡

默认 `--min-artist-count 1` 保留所有样本。报告始终给出：

- 唯一画师数、Top 20 画师；
- 长尾分布：恰好 1 张、2 张、3–5 张、6–10 张、11+ 张的画师数量；
- 稀有画师：低于 `max(2, --min-artist-count)` 张的画师数量，以及前 50 个名字/张数；
- 没有 artist 的样本数；
- 每个过滤器的丢弃数。

可选过滤：

```bash
# 去掉只有 1 张图的稀有画师
--min-artist-count 2

# 每个画师最多保留 2 张，按源文件排序稳定选择
--max-per-artist 2

# 只保留同时含 sunset 和 city lights 的样本（可重复；AND 语义）
--require-tag sunset --require-tag "city lights"
```

混合画师训练时，如果某画师只有 1–2 张图，`artist:` 条件容易过拟合到该图的构图而非画风。
建议先看报告的长尾分布，再决定是否过滤或补充数据；不要默认把所有稀有画师都删掉。

---

## 5. 文字 / 手部覆盖

报告分别统计 tag、prose 和二者的并集：

```text
text, speech bubble, sign, writing, hand, hands, fingers, holding
```

- tag 覆盖按规范化后的完整 tag 匹配；
- 英文 prose 用词边界匹配，中文 prose 用常见词（文字、对话气泡、招牌、双手、手指、手持等）匹配；
- `text_group` = `text` / `speech bubble` / `sign` / `writing` 任一；
- `hand_group` = `hand` / `hands` / `fingers` / `holding` 任一。

这些统计用于判断数据是否足以训练文字渲染和手部结构。覆盖低不等于不能训练，但意味着模型只能从
prose 的细粒度描述中学习，caption 的 short/medium 变体不会提供对应的强条件。

---

## 6. 重建缓存

预处理输出已经是 cache 的合法输入，不需要再复制图片：

```bash
moss-lora-cache \
  --input data/processed \
  --output cache/processed \
  --model models/MOSS-Image-Base \
  --target-area 1048576 \
  --max-text-length 1024 \
  --negative-prompt "worst quality, bad anatomy, watermark, jpeg artifacts, bad hands, extra fingers, missing fingers"
```

`metadata.jsonl` 里的 `file_name` 指向 `images/` 下的相对 symlink；cache 会跟随 symlink 读取源图。
如果要把数据搬到另一台机器，使用 `--copy-images` 重新生成，或手动把 symlink 替换为真实文件。

缓存完成后用 `CachedDataset` / `moss-lora-train` 读取。训练每个 step 会从该样本的
long/medium/short 变体中均匀选一个；`caption_no_artist` 只写入 metadata，当前 cache 不缓存它。

---

## 7. 分桶裁剪风险

`moss-lora-cache` 用 `build_buckets(target_area)` 生成固定 aspect 比例的 bucket，再用
`assign_bucket(width, height)` 选最近比例，最后 `fit_image` 做 cover-resize + **中心裁剪**。
裁剪比例只由宽高比决定，与图片分辨率无关：

```text
crop_fraction = 1 - (bucket_w * bucket_h) / (resized_w * resized_h)
```

报告用与 cache 相同的 `build_buckets` / `assign_bucket` / `resize` 逻辑计算每条样本的裁剪面积比例，
并列出最严重的 10 张。默认 target area 为 `1024^2`；可用 `--target-area` 改成训练配置实际使用的值。

为什么这对文字/手部重要：

- 默认 bucket 比例只到 `2:1` / `1:2`；超长条漫、竖幅海报、全景图会被裁掉大量边缘；
- 文字、对话气泡、手部经常位于画面边缘，中心裁剪可能直接删掉要训练的内容；
- 报告中的 `over_5pct` 是面积裁剪超过 5% 的样本数；`worst` 给出源文件、尺寸、bucket 和裁剪百分比。

处理建议：

1. 先看 `report.md` 的 Crop risk 表，确认最严重的图是否包含文字/手部；
2. 对极端比例的图，在**源数据之外**生成预裁剪/补边版本（不要覆盖源文件），再重新跑预处理；
3. 或按实际训练配置的 `--target-area` 重新生成报告，确认阈值；
4. 不要仅因为裁剪比例高就删除样本——先确认被裁掉的是背景还是关键内容。

---

## 8. CLI

```bash
python scripts/prepare_captions.py \
  --input data/raw \
  --output data/processed \
  --language en \
  --trigger mystyle \
  --max-tags 40 \
  --max-per-artist 8
```

需要把原始画师名映射到统一 id 时，再加 `--artist-map <你的映射.json>`（文件由你提供，仓库不带示例映射）。

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--input` | 必填 | 图片目录或 `.jsonl` |
| `--output` | 必填 | 输出目录；非空时需 `--force` |
| `--trigger` | `mystyle` | 每个变体开头 |
| `--language` | `en` | `en` / `zh` / `auto` |
| `--max-tags` | `40` | long 的 tag 上限 |
| `--min-artist-count` | `1` | 丢弃低于该数量的画师 |
| `--max-per-artist` | 无 | 每个画师最多保留张数 |
| `--require-tag` | 无 | 可重复；要求所有指定 tag 都存在 |
| `--copy-images` | 关 | 复制图片而不是相对 symlink |
| `--artist-map` | 无 | 画师名映射 JSON |
| `--tokenizer` | 自动 | 报告 token 统计用的 tokenizer；`none` 强制估算 |
| `--target-area` | `1048576` | 裁剪风险使用的 bucket target area |
| `--force` | 关 | 替换输出目录中由本脚本生成的文件 |

`--force` 只清理 `images/`、`metadata.jsonl`、`report.json`、`report.md`，不会删除输出目录中的其他文件。

---

## 9. 报告字段

`report.json` 的主要分组：

| 分组 | 内容 |
| --- | --- |
| `images` | 支持/写入/缺 sidecar/解析失败/不支持扩展/不可读/过滤计数 |
| `captions` | 写入条目数与 long/medium/short 变体数、有/无 prose 数 |
| `artists` | 唯一画师、Top 20、长尾、稀有画师、缺 artist、过滤参数 |
| `characters` | 唯一角色、Top 20、无角色样本数 |
| `tags` | 每条 tag 数分布、negative 删除计数、超过 `--max-tags` 的样本数 |
| `tokens` | long/medium/short 的 min/p50/p90/p95/p99/max、>512/1024/2048/4096 百分比、tokenizer 模式 |
| `text_hand_coverage` | tag / prose / any 的文字与手部覆盖 |
| `duplicates` | 重复图片 SHA-256、重复 long caption SHA-256 的分组统计 |
| `crop_risk` | bucket 列表、>5% 数量、p50/p95/max、最差 10 张 |
| `skipped` | 跳过原因（缺 sidecar、解析失败、不支持扩展等），不含 caption 内容 |
| `output_stats` | 输出文件数、输出目录字节数、symlink 数、引用源图字节数 |
