# Caption 去混淆方案：通用风格词剥离、真 tag-only 变体与 1k A/B

> 范围：MOSS-Image 风格 LoRA 的 caption 设计。只写方案，不改代码。
> 证据标记：`[实测]` = 本次从 pilot/cache 直接统计；`[代码]` = 仓库代码路径；`[推断]` = 基于样本/结构的估计；`[建议]` = 设计选择。

## 0. 一句话结论

v4 finalize 把 VLM prose 同时塞进 `long/medium/short`，而 prose 里的 `anime-style / digital / cel shading / soft lighting / pastel / detailed` 等词是通用模板，不是画师判别信号。**先做保守的从句级剥离，再把 `short` 恢复成真正的 tag-only，并把它的权重放在 0.25 左右**；0.5 只适合作为“最大效应”诊断臂，不适合作为生产配方。

---

## 1. 证据与问题定位

### 1.1 全量扫描（策略文档）

`/mnt/data/train_test/style_artist_strategy.md` §2.3 `[实测]`：

- 14,063 条 current 行中 **12,778 条（90.9%）**含至少一个风格词。
- 但词是弥散的：`digital` 44.0%、`line art` 39.6%、`cel shading` 33.6%、`soft shading` 26.4%、`pastel` 16.0%。
- 这些词命中的 **top-5 画师占比只有 6–9%**；top-5 画师本身约占 4.5% 图像。也就是说，风格词的画师富集度只有 1.3–2 倍，不能当 artist 标签用。

### 1.2 v4 pilot 的 caption 结构

`/mnt/data/train_test/ab_1k/final/metadata.jsonl` `[实测]`：

| 项目 | 数值 |
|---|---:|
| 总行数 | 1,000 |
| VLM valid/repaired | 727（673 ok + 54 repaired） |
| 确定性 fallback（无 VLM prose） | 273 |
| 带 `layout` 变体 | 810 |
| `long/medium/short` 三路都带 prose 的行 | 约 723（727 中 4 行因 `artist:unknown` 前缀差异未计入） |
| `layout.summary.caption` 命中保守风格词表 | 295 / 810（36.4%） |

代码路径 `[代码]`：

- `moss_lora/annotations.py:1094-1130` `build_caption_variants()`：`short=compose_caption(prefix, tags, caption.short, max_tags=SHORT_TAG_LIMIT)`。
- `scripts/finalize_dataset.py:639-647`：VLM 路径直接调用上函数；fallback 路径也把 `source_prose` 拼进 `long/medium/short`。
- `scripts/finalize_dataset.py:425-433`：`layout.summary_caption` 直接取 VLM `layout.summary.caption`，只补 prefix/tags，不剥风格词。

因此 v4 的 `short` **不是 tag-only**，只是“更短的一句 prose”。这会在第 3 节展开。

### 1.3 为什么这是 confound

- 训练时约 86% 的 step 带 prose（策略/证据文档的结构推导 `[推断]`，保守下界 73%）；纯 tag-only 只剩约 14%，而且是 40 个 tag 的长行。
- 推理时最常用的 prompt 是 `trigger + artist + 少量 general tags`。这个分布与训练分布错位：模型可以用 `anime-style, digital, soft lighting` 解释风格，不必读 `artist:<id>`。
- 结果：artist tag 的条件作用被通用风格词“短路”，尤其在中尾部画师上。

---

## 2. 通用风格词黑名单

### 2.1 三层原则

| 层级 | 用途 | 默认 |
|---|---|---|
| **C 保守（conservative）** | 只删“纯媒体/渲染/质量框架”，不碰任何可能承载内容的词；语法安全优先 | 生产默认、A/B 的 A1/A2 臂 |
| **A 激进（aggressive）** | 在 C 之上再删 medium/global 视觉属性、mood/composition；只用于消融，证明“删得更多是否更好” | 仅诊断 |
| **N 不删（content）** | 一旦删除会改变图像内容/构图/NSFW/文字的词 | 永不删 |

判断标准不是“词看起来像不像风格”，而是：**删掉后，一个没看过原图的人是否还能重建图像内容？** 不能重建的，属于内容，不删。

### 2.2 黑名单表

| 类别 | 词/短语 | 层级 | 说明 |
|---|---|---|---|
| 媒体框架 | `anime-style`, `anime style`, `anime illustration`, `anime artwork`, `anime image`, `anime scene`, `digital illustration`, `digital art`, `illustration`, `artwork`, `drawing`, `render`, `piece` | C（仅当作为句首 `A ... of` 框架） | 典型句式 `A digital anime illustration of a girl...`；删框架保留 `a girl...` |
| 渲染/技法 | `cel shading`, `cel-shaded`, `flat color`, `painterly`, `painterly style`, `soft shading`, `clean lineart`, `clean lines`, `linework`, `ink outline`, `sketch`, `watercolor`, `gouache`, `oil painting`, `3D render`, `CGI`, `pixel art`, `retro artstyle` | C（从句内）；A（全局） | `sketch/watercolor/monochrome` 可能是真实媒介，见 2.3 |
| 光照 | `soft lighting`, `warm lighting`, `cool lighting`, `dramatic lighting`, `cinematic lighting`, `volumetric lighting`, `rim lighting`, `studio lighting`, `softly lit`, `bathed in ... light` | C（纯从句）；A（句内任意位置） | 不要把 `depth of field / bokeh / lens flare / backlighting / light rays / bloom / vignetting / glitch` 当风格删；策略文档把它们归为 effect/composition，应保留为 general tag |
| 调色板 | `vibrant colors`, `muted colors`, `pastel colors`, `pastel palette`, `rich color palette`, `limited color palette`, `simple color palette`, `soft color palette`, `warm/cool color palette`, `color scheme`, `tones`, `hues`, `saturated`, `desaturated` | C（无具体颜色枚举时）；A（含枚举时改写） | `with a limited color palette of red, white, and black` 里颜色是内容；C 保留，A 改成 `in red, white, and black` |
| 质量/样板 | `masterpiece`, `best quality`, `high quality`, `highly detailed`, `ultra-detailed`, `hyper-detailed`, `intricate details`, `8k`, `4k`, `award-winning`, `trending on artstation`, `professional` | C | 这些几乎总是模板，删除风险最低 |
| 风格元话语 | `in a ... style`, `art style`, `rendered in`, `depicted in`, `reminiscent of`, `evoking`, `aesthetic`, `style` | C（`rendered/depicted in ... style`）；A（`in a ... style` 全部） | `style` 单独出现时看上下文；不要删 `hairstyle`、`style of dress` |
| 构图/氛围 | `central composition`, `dynamic composition`, `balanced composition`, `cinematic composition`, `cozy ... mood`, `serene atmosphere`, `dramatic atmosphere`, `intimate atmosphere`, `clean, intimate atmosphere` | A | 这些不是画师判别信号，但也可能影响构图/氛围；C 不删，避免误伤 |
| 类型词 | `anime`, `manga`, `comic`, `chibi`, `realistic`, `semi-realistic` | A（且仅限非内容名词前） | `anime figure` 是实体手办；`chibi` 是比例/内容；`comic` 可能是漫画页；见 2.3 |

### 2.3 内容歧义词：保守 vs 激进的关键分歧

| 词 | 为什么可能是内容 | C | A | 建议 |
|---|---|---|---|---|
| `monochrome` / `greyscale` | 图像真的是单色；是全局视觉属性，也可能是画师媒介选择 | 保留 | 删除 | 只在 tag-only 变体里保留 booru `monochrome` tag（全量覆盖 1.06%）；prose 里删掉重复描述 |
| `sketch` | 作品本身就是草图/素描；`sketchbook` 是物件 | 保留 | 只删 `sketch style/sketch rendering`，不删 `a sketch of` | 需要词性/上下文判断，纯 regex 不稳 |
| `watercolor` | 真实媒介，且对生成提示很强 | 保留 | 删除 | 策略文档：booru 支持为 0，prose 2.4%、top-5 8.6%，富集弱；可删，但先做 50 行 spot-check |
| `pastel` | 可能是真实配色（`pastel sky`、`pastel-colored garden`） | 只删 `pastel palette/colors` | 删除所有 `pastel` | C 更安全；A 会损失配色内容 |
| `detailed` | 常是内容：`a detailed view of her buttocks`、`detailed lace` | 只删 `highly detailed / ultra-detailed / detailed illustration` | 只删 intensifier | 绝不全局删 `detailed` |
| `anime` | `anime girl` 是角色呈现方式；`anime figure` 是实体手办；`anime illustration` 是媒介 | 只删句首框架里的 `anime ... illustration of` | 删 `anime/manga` 在 `girl/boy/character/scene/page/panel/collage` 前；**排除 `figure/store/convention`** | `anime figure` 是典型 false positive |
| `manga` | `manga page` 描述的是页面/分镜，属于构图内容 | 保留 | 删 `manga`，保留 `page/panel` | 激进臂会得到 `A page with three panels...`，可读但丢失“漫画页”信息 |
| `chibi` | 比例/角色设计，不是渲染风格 | 保留 | 保留 | 删了会改变角色比例 |
| `realistic` / `3D` / `CGI` | 可能是真实媒介，也可能是“画面里有 3D 物体” | 保留 | 只在 `realistic illustration/style`、`3D render` 里删 | 不要删 `3D object`、`realistic anatomy` |

### 2.4 从句级 regex（设计稿）

原则：**先按句切分，再删“从句”，不要单词级全局替换**。单词级删除最容易留下 `and a rich red color palette`、`An girl` 这类破碎语法。

```python
import re

# 1) 句首媒体框架：A digital anime illustration of a girl ... -> A girl ...
LEADING_MEDIUM = re.compile(
    r"(?i)^\s*(?:an?|the)\s+"
    r"(?:(?:highly|very|extremely)\s+)?"
    r"(?:(?:detailed|digital|anime[- ]style|anime|manga|cel[- ]shaded|"
    r"painterly|watercolou?r|3d|cgi|realistic|flat[- ]colou?red)\s+){1,5}"
    r"(?:illustration|artwork|image|drawing|render|piece)\s+of\s+"
)

# 2) 纯风格从句：rendered/drawn/depicted in a soft, painterly style
STYLE_CLAUSE = re.compile(
    r"(?i),\s*(?:rendered|drawn|depicted|painted|illustrated|shown)\s+in\s+"
    r"[^.;]{0,90}?\b(?:style|manner|technique|medium|rendering)\b"
)

# 3) lighting / palette 从句；负向前瞻保留 "palette of red, white, black" 这类内容
LIGHT_PALETTE_CLAUSE = re.compile(
    r"(?i),\s*(?:with|featuring)\s+(?:a\s+)?"
    r"(?:soft|warm|cool|dramatic|cinematic|vibrant|muted|pastel|rich|"
    r"limited|simple|pale|bold|colorful)\s+(?:[a-z-]+\s+){0,3}?"
    r"(?:lighting|colou?r palette|palette|colou?rs?|tones?)\b(?!\s+of\b)[^.;]*"
)

# 4) clean lines / soft shading 这类“技法并列”从句
TECHNIQUE_CLAUSE = re.compile(
    r"(?i),\s*(?:featuring|with)\s+(?:a\s+)?(?:blend of\s+)?"
    r"(?:clean lines?(?:\s+and\s+soft shading)?|soft shading|cel shading|"
    r"line ?art|linework|flat colou?rs?)\b[^.;]*"
)

# 5) 质量/样板词（最低风险，可全局删）
QUALITY_WORDS = re.compile(
    r"(?i)\b(?:masterpiece|best quality|high[- ]quality|ultra[- ]detailed|"
    r"hyper[- ]?detailed|highly detailed|intricate details?|8k|4k)\b,?\s*"
)

# 6) 整句纯风格：仅当整句只由“风格/元话语/停用词”组成才丢句
#    例：The scene is characterized by soft lighting and a rich red color palette.
PURE_STYLE_SENTENCE = re.compile(
    r"(?i)^(?:the\s+)?(?:scene|image|picture|artwork|illustration|composition)\s+"
    r"(?:is\s+)?(?:characterized|marked|defined)\s+by\s+(?:a\s+)?"
    r"(?:soft|warm|cool|dramatic|cinematic|vibrant|muted|pastel|rich|limited|"
    r"simple|pale|bold|colorful|clean)\s+(?:[a-z-]+\s+){0,3}?"
    r"(?:lighting|colou?r palette|palette|colou?rs?|tones?)"
    r"(?:\s+and\s+(?:a\s+)?(?:soft|warm|cool|vibrant|muted|pastel|rich|limited|"
    r"simple|pale)\s+(?:[a-z-]+\s+){0,3}?(?:lighting|palette|colou?rs?|tones?))?"
    r"\s*[.!?]?$"
)
```

实现顺序：

1. 按 `(?<=[.!?])\s+` 切句；保护引号内文本、`e.g.`、数字小数点。
2. 先匹配 `PURE_STYLE_SENTENCE`：整句纯风格才整句删。
3. 再套 `LEADING_MEDIUM`、`STYLE_CLAUSE`、`LIGHT_PALETTE_CLAUSE`、`TECHNIQUE_CLAUSE`、`QUALITY_WORDS`。
4. 清理 `,,`、`, .`、句末孤立的 `and/with/by/of`；修 `An girl -> A girl`、`A apple -> An apple`。
5. **内容 token 守卫**：删完后若句子没有任何“非停用词、非风格词”的 token，才丢句；否则保留。这样 `The scene is characterized by soft lighting with a bead of sweat on her lip` 不会被整句删掉（sweat 是内容，但子句仍可能破碎，见风险节）。
6. 只处理 prose 段，prefix + tags 原样拼回；`original` 不用 VLM prose 的规则不变。

**重要**：`PURE_STYLE_SENTENCE` 只匹配“整句都是风格”的模板。像 `The image is characterized by soft lighting and a clean anime style with subtle sweat beads on her skin.` 这种混合句，保守方案**不删**，因为删风格子串会留下 `characterized by with subtle...` 的破碎语法。这是保守方案 residual 偏高的主要原因，也是第 4 节 VLM/LLM 重写的动机。

### 2.5 真实 pilot before/after（8 条）

原文均来自 `/mnt/data/train_test/ab_1k/final/metadata.jsonl` `[实测]`；C/A 输出为设计稿/原型结果。

| # | 行/画师 | Before（原文节选） | C 保守 | A 激进 |
|---|---|---|---|---|
| 1 | `000002` deyui | `An anime-style illustration of a girl with short black hair, grey eyes, and black-framed glasses, wearing a white off-the-shoulder shirt...` | `A girl with short black hair, grey eyes, and black-framed glasses, wearing a white off-the-shoulder shirt...` | 同 C |
| 2 | `000028` elocca | `... The scene is characterized by soft lighting and a rich red color palette.` | 删除整句（纯风格模板） | 删除整句 |
| 3 | `000136` far_side | `The background consists of a pale-orange sun, grey mountains, and flying birds, rendered in a soft, painterly style.` | `The background consists of a pale-orange sun, grey mountains, and flying birds.` | 同 C；另可删 `painterly` |
| 4 | `000223` m-da_s-tarou | `The scene is set against a tiled background, featuring a blend of clean lines and soft shading.` | `The scene is set against a tiled background.` | 同 C |
| 5 | `000228` gweda | `Soft lighting and a pale color palette create a clean, intimate atmosphere.` | 删除整句 | 删除整句 |
| 6 | `000037` artist_12784 | `A monochrome manga page of a young woman with long hair and a tank top reflecting on her work while sitting in a chair.` | 不变（`monochrome/manga` 视为内容/媒介） | `A page of a young woman with long hair and a tank top...`（丢失“单色漫画页”信息） |
| 7 | `000016` chigusa_minori | `Two anime girls in school uniforms hugging against a blue background with script text.` | 不变 | `Two girls in school uniforms hugging against a blue background with script text.` |
| 8 | `000077` iumu | `An anime figure of a girl in black lace lingerie posing on a red velvet chair with a dark halo and wings.` | 不变 | **必须不变**：`anime figure` 是实体手办，`anime` 不是风格框架。这是 false-positive 守卫用例 |

补充（v1 clean30 的典型 prose，`/mnt/data/train_test/clean30_top256/metadata.jsonl`）：

- 原文：`A highly detailed anime-style digital illustration, soft warm dusk lighting, cozy serene summer evening mood, central composition framed by lush blue and white hydrangea bushes on both sides.`
- C：`An image, cozy serene summer evening mood, central composition framed by lush blue and white hydrangea bushes on both sides.`（可读，但仍保留 mood/composition）
- A/LLM 重写建议：`A cozy summer evening scene framed by lush blue and white hydrangea bushes on both sides.`

### 2.6 1k pilot 上的实测效果（原型）

在 723 条有 prose 的行上跑原型规则 `[实测-原型]`：

| variant | C 改动率 | A 改动率 | C 残留风格词行 | A 残留风格词行 | 平均删除字符（C/A，仅改动行） |
|---|---:|---:|---:|---:|---:|
| long | 22.4% | 48.7% | 55.7% | 50.1% | 41 / 36 |
| medium | 20.2% | 40.5% | 24.9% | 16.9% | 31 / 28 |
| short | 0.6% | 9.5% | 8.4% | 5.4% | 8 / 9 |

解读：

- 风格 prose 主要集中在 `long/medium` 的**句首框架**和**尾部从句**，不在 `short`。`short` 的 prose 基本是内容句（8.6% 含风格词），所以“删 short”不是去混淆的主力。
- A 的 long 残留仍有 50.1%，因为大量命中是内容用法：`detailed view of...`、`monochrome manga page`、`anime illustration vs cosplay photograph`。**纯 regex 无法在不伤内容的前提下把风格词清零**；这是第 4 节要用 VLM/LLM 重写的核心原因。
- `layout.summary.caption` 有 36.4% 命中风格词表；如果只处理 `long/medium/short`，`layout` 变体（默认 20% 权重）仍会漏风格 prose。

---

## 3. 真 tag-only 变体与权重 sweep

### 3.1 先修定义：v4 的 `short` 不是 tag-only

`moss_lora/annotations.py:1123-1124` `[代码]`：

```python
short=compose_caption(prefix, tags, caption.short, max_tags=SHORT_TAG_LIMIT)
```

`scripts/finalize_dataset.py:639-647` 对 VLM 路径和 fallback 路径都这样拼。因此：

- `--variant-weights short=0.5` 只是把“一句 prose 的短 caption”提到 50%，**不是 tag-only**。
- v4 pilot 的 `short` 有 727/1000 行带 prose；v1 clean30 的 `short` 约 96.8% 是纯 tag（证据文档 §3.4）。这就是“短 tag 条件在 v4 中消失”的原因。

三种改法：

| 方案 | 改动 | 优点 | 缺点 | 建议 |
|---|---|---|---|---|
| **A. 直接把 `short` 改成 tag-only** | `build_caption_variants` 的 `short` 传 `prose=""`；finalize fallback 同步 | 改动最小；恢复 v1 行为；5 个 variant 不变；weight 表和 `DEFAULT_VARIANT_WEIGHTS` 不用改 | 失去“短 prose”变体 | **1k A/B 用这个** |
| **B. 新增 `tag_only` kind** | `CaptionVariants` 加字段；`as_dict`/`VARIANT_ORDER`/`caption_variant_records`/`DEFAULT_VARIANT_WEIGHTS`/finalize 行都加一路 | 保留 `short` prose，可做 6 路对照 | 改 5 处，cache/index 兼容性要测；6 路权重更复杂 | 生产若确认短 prose 有价值再用 |
| **C. finalize 后处理 metadata** | 在 `row["caption"]` 写 `tag_only` | 不动 annotations | `cache._caption_records` 不读 `tag_only`，仍需改 cache | 不推荐 |

**A 的精确写法**（设计）：

```python
short=compose_caption(prefix, tags, "", max_tags=SHORT_TAG_LIMIT)
```

同时 `finalize_dataset.py` 的 fallback 分支 `short_text = compose_caption(prefix, tags, source_prose, max_tags=12)` 也要改成 `prose=""`，否则 273 条 fallback 行仍会带 source prose。

`layout.summary.caption` 也要处理：它来自 VLM，36.4% 命中风格词；建议在 `_layout_for_row` 里对 `summary_caption` 跑同一套剥离，或重写 prompt 让 summary 只描述内容。

### 3.2 权重语义（别把绝对值当百分比）

`moss_lora/cache.py:61-67` `[代码]`：

```python
DEFAULT_VARIANT_WEIGHTS = {
    "original": 0.16,
    "long": 0.64 / 3,
    "medium": 0.64 / 3,
    "short": 0.64 / 3,
    "layout": 0.20,
}
```

`normalize_variant_weights()` 对每个 item **只保留实际存在的 kind，再归一化到 1.0**；`data.variant_weights` 覆盖时，**不在 override 里的 kind 权重为 0**，全 0 才回退均匀（`cache.py:78-101`、`dataset.py:51-63`）。

pilot cache 的实际 kind 组合 `[实测]`：

| item 的 variant_kinds | 行数 |
|---|---:|
| `original, long, medium, short, layout` | 605 |
| `original, medium, short, layout`（无 long） | 202 |
| `original, long, medium, short`（无 layout） | 122 |
| `original, medium, short`（无 long/layout） | 68 |
| `original, short, layout`（无 long/medium） | 3 |

所以“给 short 加权重”并不会等比例从 long 抢：**有 27.3% 的行根本没有 long**（202 + 68 + 3），short 的权重主要从 medium/original/layout 抢。这正是为什么 0.5 会明显饿死 medium 和 layout，而不是只饿死 long。

### 3.3 Sweep 结果（用真实 index 逐 item 归一化）

下表假设 `short` 已经是真 tag-only，其余变体文本不变；权重为 `original=0.16, long=medium=0.2133, layout=0.20`，`short=w`。百分比是该 kind 在 1k 个 item 上的期望采样占比 `[实测-按 index 计算]`。

**有 layout 的 810 行：**

| short 权重 | original | long | medium | short（tag-only） | layout |
|---:|---:|---:|---:|---:|---:|
| 0.213（默认） | 17.1% | 15.9% | 22.7% | 22.8% | 21.4% |
| 0.15 | 18.4% | 17.0% | 24.4% | 17.2% | 23.0% |
| **0.25** | 16.5% | 15.4% | 21.8% | **25.7%** | 20.6% |
| 0.35 | 14.9% | 14.0% | 19.8% | 32.6% | 18.6% |
| 0.50 | 13.1% | 12.4% | 17.3% | 40.9% | 16.3% |

**无 layout 的 190 行：**

| short 权重 | original | long | medium | short（tag-only） |
|---:|---:|---:|---:|---:|
| 0.213（默认） | 22.6% | 17.1% | 30.1% | 30.1% |
| 0.15 | 24.9% | 18.6% | 33.2% | 23.3% |
| **0.25** | 21.5% | 16.4% | 28.6% | **33.5%** |
| 0.35 | 18.9% | 14.6% | 25.2% | 41.3% |
| 0.50 | 16.0% | 12.6% | 21.3% | 50.0% |

**全部 1,000 行合计：**

| short 权重 | original | long | medium | short（tag-only） | layout |
|---:|---:|---:|---:|---:|---:|
| 0.213（默认） | 18.2% | 16.2% | 24.1% | 24.2% | 17.3% |
| 0.15 | 19.6% | 17.3% | 26.0% | 18.4% | 18.6% |
| **0.25** | 17.4% | 15.6% | 23.1% | **27.2%** | 16.7% |
| 0.35 | 15.7% | 14.1% | 20.8% | 34.3% | 15.1% |
| 0.50 | 13.6% | 12.4% | 18.1% | 42.6% | 13.2% |

### 3.4 为什么 0.5 大概率太高

1. **42.6% 的 step 完全没有 prose/layout/text 条件**。tag-only 只有 `trigger + artist + character + 12–25 general tags`；布局、OCR 文字、NSFW 细节、构图描述都只存在于 prose/layout 变体里。0.5 时这些能力只得到 57.4% 的 step，且 layout 在有 layout 的行上从 21.4% 降到 16.3%。
2. **medium 被饿死**：有 layout 的行 medium 从 22.7% 降到 17.3%；而 medium 是内容密度最高、长度适中的变体（p50 约 164 tokens）。对文字/构图/NSFW 保真不利。
3. **tag-only 过拟合风险**：tag-only 是固定顺序、12–25 个短 tag；42.6% 的 step 都在这个分布上，模型容易把“短 tag 列表”本身当成条件模板，P4 全 caption 和 P5 prose-only 反而退化。
4. **0.25–0.35 已足够提供 exclusivity**：0.25 时每 3.7 步有 1 步 tag-only；0.35 时约 2.9 步 1 步。对于“让模型见过推理时的短 prompt”这个目标，27–34% 已经足够；再加只是挤压内容变体。
5. **0.5 仍有诊断价值**：如果 0.5 都不能提升 P1，说明 tag-only 分布偏移不是主因，应该回到 top128/artist-balanced sampling（H1）。所以把 0.5 作为**可选正对照**，不作为生产默认。

### 3.5 YAML 片段

训练时覆盖（不需要重建 cache；但如果 `short` 文本改成了 tag-only，必须先重建 cache）：

```yaml
data:
  cache_dir: /home/user/cache/clean30_v4_1m
  # 相对权重；不在表里的 kind 会被视为 0。
  # 每个 item 只保留实际存在的 kind，再归一化；因此这里不需要凑成 1.0。
  variant_weights:
    original: 0.16
    long: 0.2133
    medium: 0.2133
    short: 0.25      # 真 tag-only，需配合 build_caption_variants 的 short prose="" 
    layout: 0.20
```

缓存构建时也可以直接固化：

```bash
moss-lora-cache ... --variant-weights original=0.16,long=0.2133,medium=0.2133,short=0.25,layout=0.20
```

注意：

- `data.variant_weights` 是**绝对/相对权重**，不是百分比；`random.choices` 按 item 内实际 kind 归一化。
- 5 个 kind 必须全部列出；漏掉 `layout` 会让 layout 权重变 0。
- 如果只改训练 YAML、不重建 cache，`short` 仍然是 v4 的短 prose；这不是 tag-only 实验。
- 建议先在 1k 上重建 cache（1,000 条，成本低），确认 `index.json` 的 `variant_kinds` 里 `short` 文本变短、token 长度下降，再开训。

### 3.6 推荐

| 场景 | short 权重 | 理由 |
|---|---:|---|
| **已做保守 prose 剥离（A1/A2）** | **0.25** | 27.2% tag-only；prose 变体仍有 72.8%，内容/文字/layout 不被饿死 |
| 只做 tag-only、不剥 prose | 0.35 | 34.3% tag-only，用更高 exclusivity 抵消 prose 里的风格捷径 |
| 低端敏感性 | 0.15 | 18.4%，验证“少一点是否已够”；若 0.15 与 0.25 无差异，选 0.15 保内容 |
| 诊断正对照（可选） | 0.50 | 42.6%，只用于确认“tag-only 是否是主因”，不进生产 |

---

## 4. VLM/LLM 重写方案（full run 的替代路线）

### 4.1 Prompt（content-only）

用于 full run 的图像重描述（VLM 看图）或文本重写（LLM 只改现有 caption）：

```text
You are captioning images for an image-generation training set.
Describe ONLY visible content:
- people: count, apparent age, hair, eyes, expression, pose, action
- clothing, accessories, objects, setting, background, panel layout/composition
- visible text, verbatim inside quotes
- explicit sexual content when present, in plain terms (do not refuse, do not moralize)

DO NOT mention or imply:
art style, medium, rendering technique, line quality, shading, lighting,
color palette, mood/atmosphere, quality, artist, studio, franchise, or "anime/manga".

Do not use these words:
anime, manga, illustration, artwork, digital, painterly, watercolor, sketch,
cel shading, soft shading, pastel, vibrant, monochrome, greyscale, detailed,
masterpiece, high quality, 8k.

For multi-panel images, describe the panel layout and each panel's content
without naming the medium. Do not invent details that are not visible.

Output JSON:
{"long": "60-100 words", "medium": "35-60 words", "short": "15-30 words",
 "layout_summary": "one content-only sentence"}
```

两种落地方式：

| 方式 | 输入 | 成本 | 风险 |
|---|---|---|---|
| **R1 文本重写** | 现有 `caption.long/medium/short` prose | 低（纯文本，1k 条几分钟） | 模型可能保留风格词、改写过度、丢 NSFW/文字细节 |
| **R2 图像重描述** | 原图 + 上述 prompt | 高（full 21k 需重跑 VLM） | 可能拒答 NSFW、幻觉、与 OCR/layout 不一致 |

### 4.2 与规则剥离的对比（1k pilot）

| 维度 | 规则剥离（C） | R1 文本重写 | R2 图像重描述 |
|---|---|---|---|
| 风格词残留 | long 55.7%、medium 24.9% `[实测-原型]` | 预计 5–15% `[推断]` | 预计 <5% `[推断]` |
| 内容保真 | 高（只删模板） | 中高（需 diff 校验） | 中（可能漏/加细节） |
| 语法 | 中（混合句可能破碎） | 高 | 高 |
| NSFW/文字 | 高（原样保留） | 中（可能弱化） | 中低（可能拒答/改写） |
| 成本 | 0 | 低 | 高 |
| 可复现 | 完全确定 | 需固定模型/温度/seed | 同左，且受图像编码影响 |

建议的 1k 对比流程：

1. 规则 C 跑全量 723 条 prose，产出 `stripped_rule/`。
2. R1 跑同一批 723 条，prompt 固定、temperature 0（或低）、记录模型版本。
3. 抽 50 条做人工对比（见第 6 节）；同时算：
   - 风格词残留率（同一词表）；
   - 内容 token recall（人物/服装/颜色/数量/文字/NSFW 术语）；
   - 引号内文字保留率；
   - 语法错误率；
   - 平均长度变化。
4. 只有当 R1 的内容 recall ≥ 规则 C 且残留率显著更低时，才考虑 full run 用 R1；否则规则 C 更稳。
5. R2 先在 100–200 条（含 NSFW/文字/多分镜）上验证拒答率和内容 recall，再决定 full run。

**关键提醒**：即使 R1/R2 完美，`build_caption_variants` 仍会把 `caption.short` 拼进 `short`。要得到真 tag-only，仍必须改 `short` 或新增 `tag_only`（3.1 的 A/B 方案）。

---

## 5. 1k A/B 实验设计

### 5.1 三个 arm（最小可解释）

| Arm | caption | cache | short 权重 | 隔离什么 |
|---|---|---|---:|---|
| **A0 v4-control** | 现有 v4 finalize | 现有 `ab_1k/cache_1m` | 默认（0.2133） | 基线；short 仍带 prose |
| **A1 strip-prose** | 保守规则剥离 `long/medium/short` + layout summary；short 仍带 prose | 重建 | 默认 | prose 风格词的影响 |
| **A2 tag-only-0.25** | 在 A1 基础上，`short` 改真 tag-only | 重建 | **0.25** | tag-only 分布 + 去 prose 的联合效应 |

- A1 vs A0：隔离“prose 风格词”。
- A2 vs A1：隔离“真 tag-only + 0.25 权重”。
- 如果预算允许，加 **A3 tag-only-0.50** 作为诊断正对照；不要把它当生产候选。
- 三个 arm 必须用**同一批 1,000 张图、同一 cache 分辨率/桶、同一 tokenizer/max_text_length=1024**。

### 5.2 固定超参（沿用 1k pilot）

`[实测]` 来自 `/mnt/data/train_test/ab_1k/report.md`：

| 项目 | 值 |
|---|---|
| steps | 1,000（1 epoch，batch 1） |
| batch size | 1 |
| grad accumulation | 1 |
| lr | 5e-5 cosine，warmup 50 |
| rank / alpha | 64 / 64 |
| seed | 42 |
| buckets | 1M 像素，11 个桶 |
| dtype / checkpointing | bf16 / true |
| logit_normal | mean -1.09（或按 clean30 的 -0.40，但三臂必须相同） |
| sigma cutoff / uniform mix | 0.95 / 0.01 |
| EMA | 无 |

三臂只改 caption/cache/short 权重，其他全部相同。1,000 步在 5090 上约 21 分钟/arm（1.285 s/step）`[实测]`，三臂约 1 小时 + cache 重建。

### 5.3 Prompt matrix（P1–P5）

| ID | Prompt | 目的 |
|---|---|---|
| **P1 trigger only** | `@CH3OCH3, 1girl, solo, upper body` | 裸 trigger + 固定内容 scaffold；测“没有 prose 时 artist 是否可读” |
| **P2 artist only** | `artist:<id>, 1girl, solo, upper body` | 无 trigger；测 trigger 是否在替 artist 干活 |
| **P3 both** | `@CH3OCH3, artist:<id>, 1girl, solo, upper body` | trigger + artist 的组合 |
| **P4 full caption** | 从训练集取一条 `caption.long`（含 trigger/artist/tags/prose） | 完整训练分布；测 prose 是否在替 artist |
| **P5 prose only** | 只取该条的 prose（无 trigger/artist/tags） | 测“prose 单独能否决定风格” |

说明：P1–P3 加 `1girl, solo, upper body` 是为了固定内容、让图像可比；如果严格按“trigger only”不加 scaffold，P1 会生成与内容无关的图，只能看风格，不适合内容保真对比。建议报告时注明 scaffold。

### 5.4 评测集与指标

- **画师**：6 个（2 head / 2 mid / 2 tail）× 10 张 holdout（不参与训练）。
- **规模**：6 × 5 prompt × 3 seed = 90 张/arm；三臂 270 张。seed 42/123/2026。
- **主指标**：
  1. **Artist 最近质心 top-1**：DINOv2 或 CLIP image embedding；每个画师 10 张 holdout 质心；生成图对 own vs 其他画师的 cosine margin。
  2. **P1 的 top-1 与 margin**：去 prose 后 artist 是否仍可读。
  3. **P1−P3**（artist 效应）、**P4−P1**（prose 替代效应）、**P1−P2**（trigger 替代效应）。
- **次指标**：
  4. **内容保真**：角色/服装/姿势 tag 检测；OCR 文字匹配；多分镜 panel 数。
  5. **NSFW 保真**：解剖/行为术语检测或人工检查，避免“风格对了但内容崩了”。
  6. **风格泄漏**：生成图对“风格 prompt 集”（anime style / cel shading / pastel / monochrome…）vs“内容 prompt 集”的 CLIP 相似度差；或风格分类器。
  7. **固定 sigma velocity MSE**：只作 sanity check，不作为主指标（1k/1k steps 的差异通常在噪声内 `[实测]`：base 0.53393 vs 0.53369）。
- **人工**：每个生成图 vs 同画师/其他画师 holdout 三选一，盲评；至少 2 名评审，报告准确率和一致性。

### 5.5 预期与判读

| 观察 | 解释 | 下一步 |
|---|---|---|
| A1 的 P1 明显 ↑，P4 略 ↓ | prose 风格词是捷径 | 采用剥离 + 0.25 |
| A2 的 P1 再 ↑，P4−P1 gap ↓ | tag-only 分布偏移是叠加因素 | 采用 A2 配方 |
| A2 的 P4/内容保真明显 ↓ | short 权重太高，饿死内容 | 降到 0.15–0.25，或回到 A1 |
| A0/A1/A2 的 P1 都很低 | artist 频率/255 路是主因（H1） | 先 top128 / artist-balanced sampling，再谈 caption |
| P5 单独就能出风格，P1 不行 | prose 在替 artist 干活 | 必须去 prose + tag-only |

---

## 6. 风险与人工 spot-check

### 6.1 风险表

| 风险 | 具体例子 | 缓解 |
|---|---|---|
| 删掉画师特异风格 | 某画师确实只画 watercolor/painterly/monochrome；删掉后 artist tag 负担更重 | 保守 C 保留 `monochrome/sketch/watercolor`；只删 `rendered in ... style` 框架；tag-only 变体保留 booru style tag（但覆盖率仅 1–3%） |
| 删掉 NSFW/内容 | `a detailed view of her buttocks`、`detailed lace`、`with a bead of sweat` | 不全局删 `detailed`；内容 token 守卫；NSFW 术语白名单；50 行 spot-check 必含 NSFW |
| 删掉文字/构图 | 引号内台词、`six-panel`、`side-by-side comparison`、`manga page` | 保护引号/数字/panel 词；`PURE_STYLE_SENTENCE` 只对纯风格句生效；混合句不删 |
| regex false positive | `anime figure`（手办）、`3D object`、`pastel sky`、`sketchbook`、`comic store` | 上下文 guard：`anime(?=\s*(?:girl|boy|...))` 排除 `figure`；`3D` 只在 `3D render` 删；`pastel` 只在 `pastel palette/colors` 删 |
| 语法破碎 | `An girl`、`, and a rich red color palette`、`, with ...` 悬空 | 整句/整从句删除；article 修复；孤立连接词清理；30–50 条 golden tests |
| 混合句无法安全删 | `The image is characterized by soft lighting and a clean anime style with subtle sweat beads...` | 保守方案不删，留给 R1/R2 重写；不要用单词级替换硬删 |
| layout 泄漏 | `layout.summary.caption` 36.4% 命中风格词 | 对 summary 跑同一剥离；或 re-caption prompt 要求 content-only |
| 缓存不一致 | 改了 metadata 但没重建 cache；`short` 还是 prose | 重建后检查 `index.json` 的 `variant_kinds` 与 `text_lengths`；`caption_sha256` 变化会自动失效 |
| 评测过拟合 | 1k、1k steps、训练集=评测集 | 只用 holdout 做 artist 指标；固定 sigma MSE 只作 sanity；至少 3 seed |

### 6.2 50 行人工 spot-check（上线前必须）

抽样：从 1k 里分层抽 50 行 `[建议]`：

- 10 行含可见文字（OCR/引号）；
- 10 行 NSFW/露骨内容；
- 10 行多分镜/拼贴（`multi_panel`/`collage`）；
- 10 行风格词密集（long 里 ≥3 个黑名单词）；
- 10 行随机（含 fallback 行）。

每行对比 before/after，记录 4 个二值项：

1. 内容是否丢失（人物/服装/姿势/背景/NSFW/文字）——**任何丢失记 fail**；
2. 是否仍有黑名单风格词；
3. 语法是否可读（无 `An girl`、无悬空 `and`）；
4. 引号内文字是否逐字保留。

验收线 `[建议]`：

- 内容丢失 ≤ 2/50（4%）；
- 语法错误 ≤ 3/50（6%）；
- 引号文字保留 50/50；
- 风格词残留（保守 C）允许存在，但必须确认残留都是内容用法（`monochrome manga page`、`detailed view`）。

不达标：退回更保守的规则（只删句首框架 + 质量词），或改用 R1 文本重写；不要为了降残留率硬删内容。

---

## 7. 五条总结

1. **先承认 v4 的 `short` 不是 tag-only**：`build_caption_variants` 仍拼 `caption.short` prose；只调 `short=0.5` 只是“更短 prose 的 50%”，不是 tag-only。真 tag-only 必须改 `short` 的 prose 为空，或新增 `tag_only` kind。
2. **默认用保守从句级剥离，不用单词级黑名单**：句首 `A digital anime illustration of`、尾部 `rendered in ... style`、纯风格句（`soft lighting and a rich red color palette`）可安全删；`monochrome/sketch/watercolor/pastel/detailed/anime figure` 是内容歧义，保守层保留。1k 原型：long/medium 改动 20–22%，short 仅 0.6%；激进层 40–49%，但 long 仍有 50% 残留，说明纯 regex 不够。
3. **tag-only 权重选 0.25，0.35 作为“不剥 prose”的替代，0.5 只做诊断**：真实 cache 上 0.25 给 27.2% tag-only（有 layout 行 25.7%），0.35 给 34.3%，0.5 给 42.6% 且把 layout 压到 16.3%、medium 压到 17.3%，会饿死文字/构图/NSFW 内容。0.15 作为低端敏感性。
4. **A/B 用三个 arm 隔离两个因素**：A0 v4 control、A1 剥离 prose（short 仍 prose）、A2 剥离 + 真 tag-only 0.25；同 1k、同 1,000 步、同 seed 42、同 rank 64/lr 5e-5。用 P1–P5 prompt matrix + DINO/CLIP artist 质心 top-1/margin + 人工三选一 + 内容/OCR 保真判读；若 P1 在三臂都低，先查 255 路/频率，而不是继续加 caption 技巧。
5. **full run 优先 R1 文本重写，R2 图像重描述要先验证拒答/幻觉**：规则 C 免费、确定、保内容，但残留高且混合句会碎；R1 成本低、语法好，适合 21k；R2 最干净但 NSFW 拒答和内容漂移风险最高。无论哪条路线，都要处理 `layout.summary.caption`（36.4% 风格命中），并做 50 行分层 spot-check（内容丢失 ≤4%、语法错误 ≤6%、引号文字 100% 保留）。
