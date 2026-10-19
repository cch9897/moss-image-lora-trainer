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
| `layout` | 输入提供时写入；OCR `field` / `both` 模式也会写入生成的 layout，`variant` 不写 |

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

- tag 覆盖给出两列：**exact** 是规范化后的完整 tag 匹配；**family** 是词族匹配
  （`hand` 覆盖 `hand up` / `hands`、`text` 覆盖 `english text`、`holding` 覆盖 `holding cup` 等）；
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

## 8. OCR 与 layout（可选）

文字密集的图片可以额外生成官方 MOSS layout。OCR 全部在本地 CPU 运行即可，不依赖 WSL/5090。

### 8.1 安装与后端

```bash
# 轻量 CPU 后端（推荐；自带中文/英文 ONNX 模型）
pip install -e '.[ocr]'          # rapidocr-onnxruntime

# 或者使用系统 tesseract（需要对应语言包）
sudo apt install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-jpn
```

支持的后端：

| `--ocr` | 说明 |
| --- | --- |
| `none` | 默认；不跑 OCR，行为与旧版完全一致 |
| `rapidocr` | 轻量 ONNX，CPU 友好；默认模型为中文/英文，日文识别能力有限 |
| `tesseract` | 系统二进制；语言用 `--ocr-lang` 指定（`ch_sim`→`chi_sim`、`en`→`eng`、`ja`→`jpn`） |
| `easyocr` | 可选，首次使用会下载模型；支持 `ch_sim`/`en`/`ja` |
| `paddle` | 可选，首次使用会下载模型；语言映射到 PaddleOCR 的 `ch`/`en`/`japan` |

`rapidocr`/`tesseract` 不需要 GPU。`--ocr-gpu` 只是请求；如果后端或 onnxruntime 没有 CUDA provider，
会自动回退 CPU，并在报告与终端提示。大图会先缩放到 `--ocr-max-side`（默认 1600）做 OCR，再把框映射回
原图；这不会改变最终 bbox 的坐标系。2048 更慢但对小字更稳，可按需 opt-in。

`--ocr-lang ja` 只对 tesseract（需安装 `jpn` traineddata）或 easyocr 有意义；rapidocr 1.2.3
自带模型只支持中文/英文，传 `ja` 不会切换模型，日文请换后端或人工检查。

### 8.2 坐标变换

OCR 在 **EXIF 转置后的 RGB 图**上运行，与 `moss_lora.cache.fit_image` 输入一致。
对一个源图尺寸 `iw×ih`、bucket `bw×bh`：

```text
scale   = max(bw/iw, bh/ih)
new_w   = max(bw, round(iw*scale))
new_h   = max(bh, round(ih*scale))
left    = (new_w - bw) // 2
top     = (new_h - bh) // 2

x_bucket = x_original * (new_w/iw) - left
y_bucket = y_original * (new_h/ih) - top
x_center = x_bucket - bw/2
y_center = y_bucket - bh/2
```

layout item 使用 `bbox: [y1, x1, y2, x2]`（`bbox_format: "yxyx"`）和
`bbox_coord_space: "abs_centralized"`。完全在中心裁剪之外的框会被丢弃；部分越界的框会被裁到画布边界，
报告分别计入 `outside_crop` / `clipped`。这也是为什么预处理必须知道训练时的 `--target-area`：换 bucket
后 bbox 不再对应。

### 8.3 Layout JSON

```json
{
  "summary": {
    "meta": ["anime"],
    "canvas": "single",
    "caption": "mystyle, artist:alice, character:hero, 1girl, solo, a city street with a sign."
  },
  "items": [
    {
      "label": "text",
      "caption": "visible text",
      "text": "EXIT",
      "bbox": [-118, -118, -78, -28]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

`summary.caption` 在正常/预算内情况下包含 trigger、`artist:`、`character:` 和当前语言的 prose；
只有极端预算触发最后一级 `prefix-only` 时才会暂时去掉 prose，并在报告里标记 `over_budget`（此时仍保留
trigger/artist/character）。items 按从上到下、从左到右排序；相同文本且框高度重叠的识别结果会去重；
空文本被丢弃；每张图最多保留
`--ocr-max-items` 个框。OCR 原文只写入 `items[].text`，不翻译、不改写。

### 8.4 Layout 模式

| `--layout-mode` | `metadata.layout` | `caption.long` | 训练时是否看到 layout |
| --- | --- | --- | --- |
| `field` | 写入 | 普通 prose caption | 否（cache 优先读 `caption`） |
| `variant` | 不写 | 紧凑 layout JSON | 是（约 1/3 的 step，和 medium/short 均分） |
| `both` | 写入 | 紧凑 layout JSON | 是；默认 |

`variant`/`both` 只替换 long；medium/short 仍是普通 prose/tag caption，`caption_no_artist` 也不变。
这样训练时既能看到 layout 结构，也不会丢掉短 caption 的主体/动作监督。`field` 适合只把 layout
交给外部工具或人工检查，cache 本身不会训练它。

### 8.5 Token 预算与人工检查

`--layout-max-tokens`（默认 1024）限制单个 layout JSON 的 token 数。token 统计使用与
`moss_lora.text.TextEncoder.encode` 完全相同的 chat template（`apply_chat_template(...,
add_generation_prompt=True, enable_thinking=True)`）再 tokenize，因此这里的 1024 就是
`moss-lora-cache --max-text-length 1024` 的 1024。超出时按以下顺序缩短：

1. `summary.caption` 从完整 long 降为 medium，再降为“trigger/artist/character + 第一段 prose”；
2. 从置信度最低的 text item 开始删除；
3. 仍超预算时只保留 trigger/artist/character（报告 `over_budget`）。

报告会给出 layout token 的 p50/p95/max、超过 1024/2048 的比例和截断数量。OCR 对动漫字体、描边字、
竖排日文的识别质量有限；上线前必须把 bbox 画到图上人工抽查，低置信度样本建议重标或跳过。
OCR/bbox 质量差时，宁可只用 `--layout-mode field` 或 `--ocr none`，也不要把错位 bbox 送进训练。

### 8.6 重建缓存

layout 会作为普通 caption 走同一条文本编码路径。使用 `variant`/`both` 时建议把
`moss-lora-cache --max-text-length` 保持 1024（或按报告的最大 layout token 调到 2048）：

```bash
moss-lora-cache \
  --input data/processed_ocr \
  --output data/cache_processed_ocr \
  --model models/MOSS-Image-Base \
  --target-area 1048576 --max-text-length 1024 \
  --negative-prompt "worst quality, bad anatomy, watermark, jpeg artifacts, bad hands, extra fingers, missing fingers"
```

预处理阶段只加载 OCR 和 tokenizer，不加载 9B 文本编码器或 VAE。

---

## 9. CLI

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
| `--target-area` | `1048576` | 裁剪风险与 bbox 使用的 bucket target area |
| `--ocr` | `none` | `none` / `rapidocr` / `tesseract` / `easyocr` / `paddle` |
| `--ocr-lang` | `ch_sim+en` | OCR 语言，`+` 分隔；`ja` 需 tesseract/easyocr（rapidocr 仅 ch/en） |
| `--ocr-gpu` | 关 | 请求 GPU；不可用时自动回退 CPU |
| `--ocr-min-confidence` | `0.5` | 丢弃低置信度框 |
| `--ocr-max-items` | `20` | 每张图最多 text items |
| `--ocr-max-side` | `1600` | OCR 输入最大边；框会映射回原图（2048 更慢/小字更稳） |
| `--layout-mode` | `both` | `field` / `variant` / `both` |
| `--layout-max-tokens` | `1024` | 单个 layout JSON 的 token 预算 |
| `--force` | 关 | 替换输出目录中由本脚本生成的文件 |

`--force` 只清理 `images/`、`metadata.jsonl`、`report.json`、`report.md`，不会删除输出目录中的其他文件。

---

## 10. 报告字段

`report.json` 的主要分组：

| 分组 | 内容 |
| --- | --- |
| `images` | 支持/写入/缺 sidecar/解析失败/不支持扩展/不可读/过滤计数 |
| `captions` | 写入条目数与 long/medium/short 变体数、有/无 prose 数 |
| `artists` | 唯一画师、Top 20、长尾、稀有画师、缺 artist、过滤参数 |
| `characters` | 唯一角色、Top 20、无角色样本数 |
| `tags` | 每条 tag 数分布、negative 删除计数、超过 `--max-tags` 的样本数 |
| `tokens` | long/medium/short 的 cache 同款 chat-template token 数（min/p50/p90/p95/p99/max、>512/1024/2048/4096 百分比、tokenizer 模式） |
| `ocr` | 后端/设备/语言、耗时、处理图数、有文字图数、框数、裁剪外/裁剪/去重/截断、保留框的置信度分布 |
| `layout` | 模式、layout 图数/item 数、layout token 分布与 >1024/2048 比例、截断/超预算计数 |
| `text_hand_coverage` | tag exact / tag family / prose / any 的文字与手部覆盖 |
| `duplicates` | 重复图片 SHA-256、重复 long caption SHA-256 的分组统计 |
| `crop_risk` | bucket 列表、>5% 数量、p50/p95/max、最差 10 张 |
| `skipped` | 跳过原因（缺 sidecar、解析失败、不支持扩展等），不含 caption 内容 |
| `output_stats` | 输出文件数、输出目录字节数、symlink 数、引用源图字节数 |
