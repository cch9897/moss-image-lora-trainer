# OCR / layout 人工复核

OCR 只是**预填**。动漫描边字、手写体、竖排文字、艺术字经常被 RapidOCR 漏掉或认错；训练前应把
文本框和原文逐图检查一遍。只有**真的含文字**的图片需要 layout，其余图片保持普通 prose caption 即可。

`moss-lora-review` 是一个本地 FastAPI 页面：按训练 bucket 显示裁剪后的图片，允许增删/拖动/缩放
文本框、逐字修正文本、标记“无文字”，并把结果写回可被 cache 直接训练的 layout。它不加载模型、
不跑 OCR、不需要 GPU。

## 1. 安装与启动

```bash
pip install -e '.[review]'      # 或已安装 dashboard 依赖时直接使用

moss-lora-review \
  --metadata /path/to/processed_ocr/metadata.jsonl \
  --images /path/to/processed_ocr \
  --output /path/to/layout_review \
  --host 127.0.0.1 --port 7862 \
  --filter text --target-area 1048576
```

也可以不安装 console script：

```bash
python -m moss_lora.review --metadata ... --images ... --output ...
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--metadata` | 必填 | 原始 `metadata.jsonl`（或上次的 `metadata_reviewed.jsonl`） |
| `--images` | 必填 | 数据集根目录或其 `images/` 目录；路径必须在该根目录内 |
| `--output` | 必填 | 写 `metadata_reviewed.jsonl`、`review_state.json`、`review_report.md` |
| `--host` | `127.0.0.1` | 仅本机；绑定 `0.0.0.0`/`::` 会在终端警告 |
| `--port` | `7862` | HTTP 端口 |
| `--filter` | `text` | 初始队列：`text` = 需要复核的图片，`all` = 全部 |
| `--target-area` | `1048576` | 必须与训练/cache 使用的 bucket target area 一致 |
| `--fresh` | 关闭 | 忽略已有 `metadata_reviewed.jsonl`，从 `--metadata` 重新开始 |
| `--open` | 关闭 | 启动后打开默认浏览器 |

默认会从 `--output/metadata_reviewed.jsonl` 恢复（如果存在），这样重启/换浏览器不会丢进度；
要改用原始 metadata 时加 `--fresh`。输出目录并不要求与数据集同级。

## 2. 复核队列

`--filter text` 只显示两类图片：

1. **已有 layout**：需要人工确认 OCR 框位置和原文是否准确；
2. **caption 提到文字但没有 layout**：prose 或 tags 命中 `text`、`speech bubble`、`sign`、`writing`、
   `lettering`、`watermark`、`logo`、`quote`、`dialogue` 等词，或 prose 含引号原文。

标题栏和左侧显示 `remaining / queue`、`text · no layout`、已有 layout、已完成数量。全部图片可用
`All images` 切换；非队列图片也可以复核，但通常没有文字，不需要动。

左侧列表支持按文件名、画师、角色、tag、caption 搜索；圆点/文字表示 `needs review`、`edited`、
`reviewed`、`no text`。

## 3. 操作

- 在图片空白处**拖拽**画新框；拖框体移动；拖八个角/边手柄缩放。
- 右侧 `Selected box` 可编辑原文与数字 `X/Y/W/H`（bucket 像素）；`#n` 是框序号。
- `Text suggestions` 展示现有 OCR 文本，以及 caption 引号里提到的文字；点击可选中/新建框。
- `Delete box` 或 `d` 删除选中框；`n`/`p` 下一张/上一张；`s` 保存；方向键微调 1px（Shift 10px）。
- 改动后约 1.2 秒自动保存；导航前会先保存。状态栏显示 `Unsaved changes` / `Saving…` / `Saved HH:MM`。
- 图片确实没有文字时点 **Mark no text**：清空 layout、恢复 prose long caption，并记为完成。
  如果误点，可在本次会话中 **Undo no text**。保存空框列表等同于 no text。

## 4. 坐标与 bucket

页面显示的图片不是原图，而是训练时实际看到的 bucket crop。几何直接复用：

- `moss_lora.cache.build_buckets(target_area)` / `assign_bucket(width, height, buckets)`
- `moss_lora.cache.fit_image(image, bucket)`：EXIF 转置 → cover resize（LANCZOS）→ 中心裁剪

对源图 `iw×ih`、bucket `bw×bh`：

```text
scale   = max(bw/iw, bh/ih)
new_w   = max(bw, round(iw*scale))
new_h   = max(bh, round(ih*scale))
left    = (new_w - bw) // 2
top     = (new_h - bh) // 2
x_bucket = x_source * (new_w/iw) - left
y_bucket = y_source * (new_h/ih) - top
```

layout item 的 `bbox` 是整数 `[y1, x1, y2, x2]`，坐标系为 `abs_centralized`：

```text
y1 = y_bucket_top    - bh/2
x1 = x_bucket_left   - bw/2
y2 = y_bucket_bottom - bh/2
x2 = x_bucket_right  - bw/2
```

页面编辑器内部用 bucket 左上角像素 `x/y/w/h`，保存时转换回上述整数 bbox。超出 bucket 的框会被
拒绝；已有的越界 OCR 框不会显示，侧栏会提示。**改 `--target-area` 后必须重新复核**，旧 bbox 不再
对应新的裁剪。

## 5. 保存策略

默认采用与 `scripts/prepare_captions.py --layout-mode both` 相同的策略：

| 字段 | 有文本框时 | 标记 no text 时 |
| --- | --- | --- |
| `layout` | 写入完整 layout 对象 | 删除 |
| `caption.long` | 紧凑 layout JSON（cache 会训练） | 恢复原始 prose |
| `caption.medium` / `short` | 保持不变 | 保持不变 |
| `caption_no_artist` | 保持不变 | 保持不变 |
| `layout.summary.caption` | 保留 trigger/`artist:`/`character:`/prose | — |

`caption.long` 必须写 layout JSON，因为 `moss_lora.cache` 在存在 `caption` 对象时优先读
`caption.long`，不会单独读取 `metadata.layout`。只写 `layout` 字段会导致“复核了但训练看不到”。
`summary.caption` 会保留原 layout 的 summary；没有 layout 时使用 long prose，并补齐缺失的
`artist:` / `character:` / trigger。

所有写入都是原子的（临时文件 + `os.replace`）：

- `metadata_reviewed.jsonl`：保留全部行与未知字段，每行一行紧凑 JSON；
- `review_state.json`：每个 item 的 `edited` / `reviewed` / `no_text`、框数、时间；
- `review_report.md`：复核队列、剩余、edited、no text、文本框总数等统计。

复核完成后重建缓存。`moss-lora-cache` 只认 `metadata.jsonl`，所以把 reviewed 文件放进一个轻量
目录（图片用软链，不复制数据，也不改原数据集）：

```bash
mkdir -p /path/to/review_dataset
ln -sfn /path/to/processed_ocr/images /path/to/review_dataset/images
cp /path/to/layout_review/metadata_reviewed.jsonl /path/to/review_dataset/metadata.jsonl

moss-lora-cache \
  --input /path/to/review_dataset \
  --output /path/to/cache_reviewed \
  --model models/MOSS-Image-Base \
  --target-area 1048576 --max-text-length 1024
```

`--target-area` 与复核时保持一致；`metadata.jsonl` 里的 `file_name` 是 `images/...`，因此软链应保持
同样的相对布局。不要直接用 `--input /path/to/layout_review`，那个目录没有图片。

## 6. HTTP API

页面自身只使用以下本地接口（无 CORS、无外部资源）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 单文件 HTML/CSS/JS |
| GET | `/api/items?filter=text\|all&q=` | 队列摘要、计数、搜索 |
| GET | `/api/items/{index}` | caption、layout、bucket 几何、建议文本 |
| GET | `/api/image/{index}` | 该 index 的 bucket-crop JPEG |
| POST | `/api/items/{index}/save` | `{"boxes":[{"bbox":[y1,x1,y2,x2],"text":"..."}],"no_text":false}` |
| POST | `/api/export` | 强制重写 metadata/state/report |
| GET | `/healthz` | 存活检查 |

## 7. 安全

- 默认只绑定 `127.0.0.1`；没有认证，不应直接暴露到公网/不可信网络。
- 浏览器只能按 metadata index 取图，不能传任意文件路径；`file_name` 必须是相对路径、不能含
  `..`/反斜杠/绝对路径，且必须在 `--images` 根目录内。扩展名必须是图片。
- 为了兼容数据集的相对软链，图片软链会被跟随，但服务端不会用请求参数拼接文件路径，也不会提供
  metadata/state 等任意文件下载。
- 不加载模型/GPU，不访问网络，不下载权重。

## 8. 测试

```bash
pytest -q tests/test_review.py
pytest -q
ruff check .
```

测试使用 `tmp_path` 生成的合成图片/文本，不包含样本数据。若本机有 Chrome/Playwright，可额外做
页面加载与一次保存的 headless smoke；没有时至少确认 HTML/JS 结构、接口测试和手动浏览器检查。

复核输出应写在数据集之外（例如 `/path/to/layout_review/`），不要提交到仓库。
