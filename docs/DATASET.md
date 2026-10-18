# 数据集、Layout 与 Caption

本文说明 MOSS-Image LoRA 训练的数据组织、分辨率/分桶策略，以及如何用官方 caption prompt 生成高质量训练描述。

数据质量对最终效果的影响通常**大于** rank 和 learning rate。尤其是文字渲染与 layout 能力：caption 里少写一行字、bbox 坐标错位，模型就会学到错误的条件映射。

---

## 1. 支持的目录结构

### 1.1 图片 + 同名 `.txt`

```text
data/
└── images/
    ├── 0001.jpg
    ├── 0001.txt
    ├── 0002.png
    └── 0002.txt
```

`0001.txt` 是 UTF-8 文本，内容即 caption。支持 `.jpg/.jpeg/.png/.webp/.bmp`。

### 1.2 图片 + `metadata.jsonl`

```text
data/
├── images/
│   ├── 0001.jpg
│   ├── 0002.png
│   └── 0003.png
└── metadata.jsonl
```

`metadata.jsonl` 每行一个 JSON 对象，`file_name` 相对 JSONL 所在目录：

```jsonl
{"file_name": "images/0001.jpg", "caption": "A rain-soaked night market street in Taipei..."}
{"file_name": "images/0002.png", "caption": {"long": "...", "medium": "...", "short": "..."}}
{"file_name": "images/0003.png", "layout": {"summary": {"meta": ["poster"], "canvas": "single", "caption": "..."}, "items": [{"label": "text", "caption": "...", "text": "EXPLORE", "bbox": [-416, -384, -288, 384]}], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}}
```

示例见 [`examples/metadata.example.jsonl`](../examples/metadata.example.jsonl)。

### 1.3 直接传 JSONL

```bash
moss-lora-cache --input annotations.jsonl --output cache/1m ...
```

图片路径相对 JSONL 文件所在目录解析。`--input` 同时接受目录和单个 `.jsonl`。行内没有 caption 字段时，会回退到图片同名的 `.json`（优先）或 `.txt` 文件，与目录模式的 sidecar 规则一致。

### 1.4 字段优先级

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `file_name` | string | 必填；相对路径 |
| `caption` | string 或 object | 字符串 = 单一 caption；object = `{long, medium, short}`；缺省时回退到同名 `.json`/`.txt` |
| `text` | string | `caption` 的别名（JSONL 简写） |
| `layout` | object | 官方 layout JSON；与 `caption` 二选一 |

同一张图同时有 `.txt` 和 `metadata.jsonl` 时，以 manifest 为准。不要同时维护两份，容易漂移。

### 1.5 缺失 / 空 caption

缓存把以下情况都视为 **没有 caption**：

- 图片没有同名 `.json` / `.txt`（或两者都是空文件 / 纯空白）；
- sidecar 是 `{}`、`{"image_description": "   "}` 之类没有可用文本的对象；
- `metadata.jsonl` / `metadata.csv` 中的 caption 为空，且没有可用的同名 sidecar。

sidecar `.json` 或 `.txt` 除 `caption`/`text`/`prompt`/`layout` 外，也接受标注工具常见的 schema；解析按内容而不是后缀，所以 `.txt` 里同样可以放下面的 JSON（有些标注数据集就是这种布局）：

```json
{"year": 2025, "preference_level": "best", "image_description": "a red bicycle leaning against a brick wall", "extra_tags": ["bicycle"]}
```

其中 `image_description` 会作为单一 caption 使用（`extra_tags` 不单独拼接，避免与描述重复）。`metadata.jsonl`/`metadata.csv` 行和直接传入的 JSONL 也接受同一字段。

目录模式与 JSONL 模式的缺失 caption 行为一致，都由 `--missing-caption` 控制。

默认行为是 `--missing-caption skip`：为每张坏图打印 `warning: skipping <path>: ...`，不编码、不写缓存文件、不进入 `index.json`，最后打印 `N images, M skipped (no/empty caption)`。缓存 ID 只对保留的图片按排序后连续编号，因此相同输入 + 相同跳过规则可以稳定 resume；因跳过而遗留的旧 `item_*.safetensors` 会被清理。

`--missing-caption error` 在第一张坏图处失败，错误信息包含图片绝对路径、sidecar 命名提示和 `--missing-caption skip` 建议。若所有图片都被跳过，会直接报错而不是生成空缓存。

> 跳过只用于让长任务不被个别坏样本中断，**不是质量策略**。空 caption 会退化成无条件生成；训练前应修补 sidecar，或用官方 caption prompt 重新生成。

---


## 2. Caption 变体：long / medium / short

官方 caption prompt 输出三个字段：

```json
{
  "caption": {
    "long": "a detailed image description, including all non-watermark visible text",
    "medium": "a one-paragraph image description",
    "short": "a one-sentence image description"
  },
  "watermark": [],
  "borders": ["none"]
}
```

本项目的行为：

- `caption` 为 object 时，**三个非空变体都缓存**；训练每个 step 对每张图等概率选一个（对齐技术报告的 1:1:1 short/medium/long recaption 混合）。
- `caption` 为字符串时只有一个变体。
- `layout` 只有一个变体（紧凑 JSON 字符串）。
- 变体选择是 **per-sample per-step**，不是固定一张图一个 caption。这样同一张图会在不同粒度上被监督。

为什么保留 short：短 caption 迫使模型学习主体/动作，避免过度依赖长描述的措辞；long 提供文字、布局、风格细节。只保留 long 容易让模型对固定句式过拟合。

---

## 3. Layout JSON

MOSS-Image-Base 支持结构化 layout 输入（官方标注为 experimental）。格式：

```json
{
  "summary": {
    "meta": ["poster"],
    "canvas": "single",
    "caption": "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top."
  },
  "items": [
    {
      "label": "text",
      "caption": "Large dark-blue uppercase title centered at the top.",
      "text": "EXPLORE",
      "bbox": [-416, -384, -288, 384]
    },
    {
      "label": "object",
      "caption": "A flat blue mountain illustration in the lower center.",
      "bbox": [-192, -288, 384, 288]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

### 3.1 字段

| 字段 | 说明 |
| --- | --- |
| `summary.meta` | 内容类型标签，例如 `["poster"]` |
| `summary.canvas` | 画布模式，官方示例为 `"single"` |
| `summary.caption` | 整体场景/风格描述 |
| `items[].label` | `"text"` 或 `"object"` |
| `items[].caption` | 该元素的描述（位置、颜色、字体、材质等） |
| `items[].text` | 仅 `label == "text"` 时使用；要渲染的**原文** |
| `items[].bbox` | `[y_min, x_min, y_max, x_max]`，整数像素 |
| `bbox_format` | 固定 `"yxyx"` |
| `bbox_coord_space` | 固定 `"abs_centralized"` |

### 3.2 坐标系

`abs_centralized`：以画布中心为原点，x 向右、y 向下，单位是像素。对宽 W、高 H 的画布，取值范围是 `[-W/2, W/2]` 和 `[-H/2, H/2]`。

从常见的左上角坐标 `(x1, y1, x2, y2)` 转换：

```
bbox = [y1 - H/2, x1 - W/2, y2 - H/2, x2 - W/2]
```

例：1024×1024 画布上，文字框左上 `(128, 96)`、右下 `(896, 224)`：

```
bbox = [96 - 512, 128 - 512, 224 - 512, 896 - 512]
     = [-416, -384, -288, 384]
```

这正是官方示例中的 title bbox。

### 3.3 训练时的处理

`layout` 对象会被序列化为紧凑 JSON 字符串（无多余空格），再走与自然语言相同的 chat template / 文本编码路径。`items[].text` 的原文必须保留，不要翻译、不要改写大小写、不要动换行。

### 3.4 坐标必须匹配最终画布

这是 layout 数据最容易出错的地方：

- 缓存阶段会按 bucket **缩放 + 中心裁剪**。裁剪会改变画布原点与可见范围。
- 如果原图 1024×1024，但 bucket 选择了 1216×832，原来的 bbox 就不再对应。
- **建议**：layout/文字密集数据固定为正方形单 bucket（例如全部 1024×1024），或预先按目标 bucket 渲染/裁剪，再生成 bbox。
- 生成 caption 时就把画布尺寸写进 prompt（例如"on a 1024x1024 canvas"），并要求 VLM 按该尺寸输出坐标。
- 上线前把 bbox 画到图上人工检查；错位的 layout 比没有 layout 更伤模型。

---

## 4. 分辨率、面积与分桶

### 4.1 模型约束

- 宽高必须是 **16 的倍数**（VAE 8× + patch 2×2）。
- 官方推荐 1M / 1.5M / 2M 总像素；支持 0.25M（512²）到 4M（2K）外推。
- 图像 token 数 `N = (H/16) × (W/16)`；面积相同、aspect 不同时 token 数基本一致。

### 4.2 缓存分桶

`moss-lora-cache` 的参数：

| 参数 | 含义 | 建议 |
| --- | --- | --- |
| `--target-area` | 目标像素面积 | 1048576（1M）、262144（512²）、1478656（~1.5M）、2097152（2M） |
| `--max-text-length` | caption token 截断上限 | 默认 1024；短 caption 可用 512 降成本；layout/长文本 1024–2048 |
| `--latent-dtype` | latent 存储精度 | `fp32`（默认，推荐）或 `bf16`（缓存减半，轻微精度损失） |

bucket 边长由实现统一对齐到 64（模型只要求 16），范围 512–2048，覆盖 1:1、4:3、3:4、16:9、9:16、3:2、2:3、5:4、4:5、2:1、1:2 等常见比例。实现会生成一组面积接近 `target-area` 的 bucket，训练时按 bucket 组 batch，同一 batch 内不 padding。

### 4.3 分辨率选择

| 目标 | 推荐面积 | 典型尺寸 | 说明 |
| --- | --- | --- | --- |
| 快速迭代 / 风格 | 0.25M | 512×512 | 最快；适合先验证 caption 和数据 |
| 默认 | 1M | 1024×1024 | 对齐官方推荐；质量/成本平衡 |
| 高分辨率 | 1.5M | 1216×1216（1.48M）或 1280×1280（1.64M） | 文字更清晰；显存/时间上升 |
| 高分辨率 | 2M | 1408×1408（1.98M）或 1472×1472（2.17M） | 需要更大显存；1536² 已到 2.36M |
| 测试 | 0.25M | 512² | 只用于跑通流程 |

> 改 `--target-area` 后必须**重建缓存**；latent 与最终画布绑定。

### 4.4 裁剪

常见的预处理是"等比缩放到覆盖目标 bucket，再中心裁剪"。这对摄影/商品图通常没问题，但：

- 人脸、文字、logo 不要贴边；
- 竖图/横图混排时，主体可能被裁掉；
- 如果数据本身构图很重要，预先手工裁剪成目标 aspect，避免训练时随机裁掉关键内容。

---

## 5. Caption 质量规则

### 5.1 必须做到

1. **只描述图里可见的内容**。不要写"可能""应该是"或训练集标签里没有依据的推断。
2. **文字逐字保留**。包括中英文、大小写、标点、换行；不要翻译、不要罗马化、不要纠正拼写。
3. **描述排版**：文字的行数、顺序、相对位置、对齐、字号层级、字体风格、颜色、间距。
4. **描述构图**：主体位置、视角、景别、前中后景、留白。
5. **描述风格与光线**：摄影/插画/3D/海报；光源方向、色温、对比度、材质。
6. **不要描述水印**，也不要让水印文字进入 caption；水印只记录在 `watermark` 字段。
7. **caption 不能为空**。默认缓存会跳过空 caption 并告警（`--missing-caption skip`），但这张图就不会参与训练；请补齐 caption，不要依赖跳过。

### 5.2 好/坏示例

差：

```text
a beautiful poster, high quality, masterpiece, 1girl, blue hair
```

问题：没有构图、没有文字内容、没有风格信息；标签堆砌不能教会模型这张图独特在哪里。

好：

```text
A square travel poster on cream paper. At the top, large dark-blue uppercase sans-serif text reads "EXPLORE", centered with wide letter spacing. Below it, a flat blue mountain illustration occupies the lower half, with three overlapping peaks and a thin white snow line. A small orange sun sits in the upper right. Minimalist vector style, no gradients, generous negative space.
```

### 5.3 语言与标签顺序

- 官方推荐 **English**（训练数据最多）；中文和其他语言也支持。混合语言数据时，caption 语言尽量与目标使用场景一致。
- 如果使用 tag 序列，官方推荐顺序：**Quality → Score → Meta → Year → Character → Series → Artist → General**。
- 自然语言与 tag 可以混用，但同一数据集内保持风格一致，避免模型学到"某种写法 = 某种质量"的伪相关。

### 5.4 文字渲染数据

- 在 caption 中把文字放在引号里，逐字写出；多行用 `\n` 或真实换行。
- 明确文字的位置、字号、颜色、字体风格（例如 "handwritten Chinese in black marker"）。
- 需要精确控制位置时用 layout JSON，而不是只靠自然语言。
- 文字密集图建议 1.5M–2M 像素训练；1M 下小字容易糊。

---

## 6. 自动 Caption

### 6.1 官方 prompt

官方 User Guide 提供了一个 image recaptioner prompt，要求输出：

- `caption.long/medium/short`
- `watermark`（归一化 0–1000 的 `box_2d`）
- `borders`

完整 prompt 已**逐字**保存在 [`docs/caption_prompt.txt`](caption_prompt.txt)。来源：[MOSS-Image USERGUIDE.md — Image Captioning](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#image-captioning)。

使用时把 `{{reference}}` 替换为原始 caption（没有就留空字符串）。prompt 明确要求"Treat the original caption as a reference. Correct any claims that are not supported by the image."，所以原始 caption 只作为参考，不作为真值。

### 6.2 推荐 VLM

官方测试过的 caption 模型：

- **Gemma 4**
- **Qwen3.5 27B**
- **Qwen3.5 35B-A3B**（MoE）

官方建议开启 **thinking mode**，并给出请求参数：

```python
body = {
    "model": model_name,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }],
    "n": 1,
    "stream": False,
    "max_tokens": 16384,
    "temperature": 1,
    "top_p": 0.95,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
}
```

`temperature=1` 看起来高，但这是官方给出的 caption 请求配置；它增加描述的多样性。若同一张图需要稳定复现，可降低 temperature，但不要把它当成"更准确"的保证。

### 6.3 后处理

1. 解析 JSON；失败时重试或丢弃，不要把 Markdown 包裹的文本直接塞进 caption。
2. 检查三个 caption 字段非空；`watermark` 不能出现在 caption 文本里。
3. 用 `watermark.box_2d` 决定是否裁掉/遮挡水印，而不是让模型"忽略"水印。
4. 人工抽检 5–10%，重点看文字是否逐字正确、bbox/排版描述是否可信。
5. 保留 `long/medium/short` 三个变体；不要只留 long。

### 6.4 提升 caption 质量的技巧（官方建议）

- 加入相关世界知识（人物、作品、品牌、字体）。
- 用 agentic pipeline：先检测文字/布局，再让 VLM 在结构化中间结果上写 caption。
- "Better understanding helps improve generation"：caption 模型看得越准，生成模型学得越好。

---

## 7. 数据规模

| 目标 | 图片数 | 参考步数（1M，batch 1） | 说明 |
| --- | --- | --- | --- |
| 风格/滤镜 | 20–100 | 500–1500 | 容易过拟合；rank 16–32，lr 5e-5–1e-4 |
| 人物/角色 | 500+ | 2–5k | 多角度、多表情、多背景；caption 要区分不变特征与可变特征 |
| 领域/产品 | 500+ | 2–5k | 产品细节、材质、文字/logo 要准确 |
| 稳健概念 | 1k–5k | 3–10k | 需要更均衡的分布和更低的 lr |

更准确的步数算法：

```
steps ≈ epochs × N_images / batch_size
```

例如 200 张图、batch 1、目标 10 epoch → 2000 步。小数据集不要盲目堆步数；先看 holdout。

### 7.1 验证集

- 留出 **4–8 张**（或 4–8 个 prompt/样本对）作为验证观察集，写进 `logging.sample_val_ids`。id 使用缓存 item id（`item_00000`、`item_00001`、…），见 `cache/index.json`。
- **注意**：当前实现只把这些 id 用于验证出图，`BucketBatchSampler` 仍会从完整缓存中采样，因此它们实际上仍参与训练。若要得到真正未参与训练的 holdout，需把这些图从训练缓存/数据中排除（或单独建训练缓存），训练前先用 base 模型出基线图。
- 每 250–500 步用相同 seed/参数出图；同时看 raw 与 EMA。
- 如果验证图变差而 train loss 继续降，优先降步数/rank/lr，而不是继续训练。

### 7.2 去重与平衡

- 近重复图（连拍、同图不同裁剪）会让模型记住像素而不是概念；做感知哈希/embedding 去重。
- 主体、背景、构图、文字密度尽量均衡；否则模型会偏向多数类。
- 不要在数据里混入明显 AI 伪影/水印图，除非这就是目标域。

---

## 8. 缓存与数据变更

缓存与以下参数绑定：

- 图片内容与预处理（resize/crop）；
- `--target-area`、`--latent-dtype`；
- `--max-text-length`；
- tokenizer revision / chat template；
- `--negative-prompt`。

任何一项变化都需要重建对应缓存。缓存条目按图片+参数哈希命名；重复执行会复用已有的有效条目。

磁盘估算：

| 分辨率 | latent/图 | 文本/图 | 1k 图合计 |
| --- | --- | --- | --- |
| 512² | 0.5MB | 1–4MB | 1.5–4.5GB |
| 1024² | 2.0MB | 1–4MB | 3–6GB |
| 1536² | 4.5MB | 1–4MB | 5.5–8.5GB |

文本体积取决于 caption 长度与变体数量；layout JSON 通常比自然语言长。

---

## 9. 常见数据错误

| 现象 | 原因 | 修复 |
| --- | --- | --- |
| 训练 loss 正常但出图无 prompt 响应 | caption 为空 / 全是一个词 | 检查缓存前的文本；补 caption |
| 文字总是错字/漏行 | caption 没有逐字写；或训练分辨率太低 | 用 layout + 高分辨率；caption 原文照抄 |
| layout 不生效 | bbox 顺序写成 xyxy；或用了相对坐标 | 改成 `yxyx` + `abs_centralized` |
| layout 位置偏移 | 缓存裁剪改变了画布 | 固定 bucket/分辨率；重新生成 bbox |
| 风格过拟合、构图僵化 | 数据太单一 / 步数太多 / 只有 long caption | 增加多样性；保留 short/medium；降步数 |
| 画面出现水印 | caption 里写了水印，或数据本身有水印 | 从 caption 移除；裁掉/遮挡水印 |
| OOM 且与 batch 无关 | 文本太长（layout JSON 数千 token） | 降 `max-text-length` 或精简 layout |
| 缓存后训练报形状错误 | 改了 `target-area` 但没重建缓存 | 重建缓存，保持参数一致 |
