# 画师均衡采样设计（clean30 top256，bs8）

日期：2026-11-10  
范围：`/mnt/data/train_test/clean30_top256`、`moss_lora/dataset.py`、`moss_lora/trainer.py`、`moss_lora/config.py`、`docs/bs8_5ep_launch.md`  
结论先行：**先做“只补尾部到中位数 74”的 repeat-map 实验，不要直接全均衡。** 期望每 epoch 24,598 行、约 3,070 step/epoch、4 epoch 约 12,280 step；画师标签曝光尾部约提升 2.6 倍，头部完全不动。

---

## 0. 5 条结论

1. **推荐第一实验：upsample-only，目标 = 中位数 74。** 127 个画师被补样，增加 3,645 行/epoch，总行数 24,598，约 3,070 step/epoch（精确值需远端 `index.json` 复算），4 epoch 约 12,280 step；头部 18 个画师一个样本都不删。
2. **补到均值 79 的代价明显更高。** 140 个画师、+4,304 行、25,257 行/epoch、约 3,152 step/epoch，4 epoch 约 12,610 step（+20.6% 计算量），相对 74 只多补 13 个画师，不建议作为第一枪。
3. **全均衡到 74 会删掉 4,957 行头部样本/epoch，净减 1,312 行。** 255×74 + 771 unknown = 19,641 行，约 2,451 step/epoch；4 epoch 反而少 6.2% 计算量。头部每张图从 4 次曝光降到约 1 次，风格保真风险最大。
4. **实现走 sampler 级 repeat map，不改数据、不重缓存。** 每张图基础重复 `q=floor(T/c)`，余数 `r=T mod c` 的头 `r` 张每 epoch 轮换 +1；单画师每 epoch 精确贡献 T 张。最大重复 3 次；74 目标下 47 个画师（28–36 张）需要 3×，80 个画师（37–73 张）只需 2×。
5. **必须重算 `max_steps` 并做 A/B。** 只补中位数时若仍用 `max_steps=10,456`，实际只有 3.41 epoch；4 个完整 epoch 应设 12,279/12,280。A/B 固定总 step 预算、固定 seed 42（正式跑 3 个 seed），按 head/mid/tail 分层看 artist top-1、margin、风格相似度和记忆化指标。

---

## 1. 数据事实与口径修正

### 1.1 实际行数

`metadata.jsonl`：**20,953 行**。  
其中：

| 类别 | 画师数 | 行数 | 占比 |
|---|---:|---:|---:|
| 有名画师 | 255 | 20,182 | 96.32% |
| `unknown` | 1 个计数桶 | 771 | 3.68% |
| 合计 | — | **20,953** | 100% |

`report.json` 的 `artist_counts` 是**去重前**的 20,959 行口径：`unknown=777`，255 个有名画师合计 20,182。去重只删掉了 6 行 `unknown`，所以：

- 255 个有名画师的计数**可以原样使用**；
- `unknown` 必须用 **771**，不能用 777；
- 不要用 `report.json` 的 `sum(artist_counts)=20,959` 当训练总数。

### 1.2 分布

有名画师 255 个：

| 统计量 | 值 |
|---|---:|
| 中位数 | **74** |
| 均值 | **79.145… ≈ 79.1** |
| 最小 | 28 |
| 最大 | 298（mignon） |
| ≥150 | 18 个有名画师（若把 `unknown=777` 当桶则 19 个） |
| <74 | 127 个，合计 5,753 行 |
| =74 | 1 个 |
| >74 | 127 个 |
| <79 | 140 个，合计 6,756 行 |
| =79 | 2 个 |
| >79 | 113 个 |

分层口径（全文统一）：

| 分层 | 定义 | 画师数 | 原始行数 | 原始占比 |
|---|---|---:|---:|---:|
| tail | `<74` | 127 | 5,753 | 27.5% |
| mid | `74–149` | 110 | 11,080 | 52.9% |
| head | `≥150` | 18 | 3,349 | 16.0% |
| unknown | — | 1 桶 | 771 | 3.7% |

> 注：`unknown` 不是画师，不参与均衡；它保持 1 次/epoch。任何“每画师 T 张”的公式都只作用于 255 个有名画师。

---

## 2. 目标与总量公式

设：

- 画师 `a` 的图片数 `c_a`；
- 目标 `T`（本设计取 74 或 79）；
- `U = 771` 为 unknown 行数；
- `A = 255` 为有名画师数。

### 2.1 Upsample-only（只补尾部）

对每个画师：

$$
T_a = \max(c_a, T)
$$

总行数：

$$
N_{\text{up}}(T) = U + \sum_{a=1}^{A} \max(c_a, T)
$$

被补画师集合 `{a | c_a < T}`；补样行数：

$$
E_{\text{up}}(T) = \sum_{c_a<T} (T - c_a)
$$

### 2.2 Full equalize（全均衡）

每个画师每 epoch 正好 `T` 张；头部只保留 `T` 张（轮换），尾部补到 `T`：

$$
N_{\text{eq}}(T) = U + A \cdot T
$$

头部删除行数：

$$
D(T) = \sum_{c_a>T} (c_a - T)
$$

尾部补样行数仍为 `E_up(T)`，净变化 `E_up(T) - D(T)`。

### 2.3 单画师重复分配（核心）

对 `c_a < T`：

$$
q_a = \left\lfloor \frac{T}{c_a} \right\rfloor,\qquad
r_a = T \bmod c_a
$$

- 每张图基础重复 `q_a` 次；
- 余数 `r_a` 张图各再 +1 次；
- 该画师每 epoch 精确贡献：

$$
c_a q_a + r_a = T
$$

头部在全均衡模式下 `c_a > T` 时：

$$
q_a = 0,\qquad r_a = T
$$

即每 epoch 只抽 `T` 张，按 epoch 轮换；upsample-only 模式下头部 `q_a=1, r_a=0`，完全不删。

**最大重复次数**：

$$
\max_a \left\lceil \frac{T}{c_a} \right\rceil
= \left\lceil \frac{T}{28} \right\rceil
$$

- `T=74`：最大 **3**；`c=28…36` 的 **47 个画师**需要 3×，`c=37…73` 的 **80 个画师**最多 2×，`c=74` 为 1×。
- `T=79`：最大仍是 **3**（`c≤39` 才可能到 3×）。

示例（`T=74`）：

| 画师图片数 c | q | r | 每 epoch 组成 | 单图重复 |
|---:|---:|---:|---|---|
| 28 | 2 | 18 | 18 张 ×3 + 10 张 ×2 | 2–3 |
| 30 | 2 | 14 | 14 张 ×3 + 16 张 ×2 | 2–3 |
| 37 | 2 | 0 | 37 张 ×2 | 2 |
| 40 | 1 | 34 | 34 张 ×2 + 6 张 ×1 | 1–2 |
| 50 | 1 | 24 | 24 张 ×2 + 26 张 ×1 | 1–2 |
| 73 | 1 | 1 | 1 张 ×2 + 72 张 ×1 | 1–2 |
| 74 | 1 | 0 | 全部 ×1 | 1 |
| 298（equalize） | 0 | 74 | 每 epoch 轮换 74 张 | 0–1 |

### 2.4 确定性轮换（避免永远重复同几张）

不要每 epoch 都选“文件名前 r 张”。按稳定哈希排序后轮换：

```python
rank_key = blake2b(f"{seed}:{artist}:{image_sha256}", digest_size=8).digest()
ordered = sorted(artist_items, key=rank_key)
start = (epoch * r) % c
selected = {ordered[(start + k) % c] for k in range(r)}
```

等价的 O(1) 判定（推荐写进 sampler）：

```python
n = q + (1 if r and (rank - epoch * r) % c < r else 0)
```

性质：

- `rank` 是图片在哈希序中的下标，`0 ≤ rank < c`；
- 每个 epoch 每个画师仍然精确 `T` 张；
- 不依赖 Python 进程内 `hash()`，跨进程、跨 resume 完全一致；
- `r=0` 时无额外项，头部/整除数画师不受影响；
- 每 epoch 的额外图片会平移，28 张画师的某张图不会 4 个 epoch 都吃 +1。

> 不建议用 `random.Random(seed+epoch).shuffle(artist_items)`：需要每 epoch 对每个画师排序，且与 `dataloader` worker 的 RNG 无关但实现更重；上面的哈希序 + 平移是确定且廉价的。若坚持随机序，必须使用 `blake2b(epoch|sha)` 作为 key，不能使用全局 `random`。

### 2.5 4 epoch 的曝光不对称

以 `T=74` 为例：

- 28 张画师：每 epoch 74 次浏览 ×4 = **296 次**，平均每张 **10.6 次**，最多 12 次；
- 298 张的 mignon（若全均衡）：每 epoch 74 ×4 = **296 次选择**，按 `start=(e*74)%298` 轮换，4 epoch 覆盖 **296/298 张**，几乎每张只出现 **1 次**；
- upsample-only 的头部：每张 **4 次**（baseline 不变）。

这就是“tail 反复看、head 看四次”与“全均衡 tail 反复看、head 看一次”的本质差别。

---

## 3. 数量表与 step 估算

### 3.1 行数表（精确计数）

| 方案 | 目标 T | 被补画师 | 补样行 | 被削画师 | 删除行 | 净变化 | 新行/epoch | 相对 baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | — | 0 | 0 | 0 | 0 | 0 | **20,953** | 100% |
| upsample-only | 74 | **127** | **+3,645** | 0 | 0 | +3,645 | **24,598** | **117.4%** |
| upsample-only | 79 | 140 | +4,304 | 0 | 0 | +4,304 | **25,257** | **120.6%** |
| full equalize | 74 | 127 | +3,645 | 127 | −4,957 | **−1,312** | **19,641** | **93.7%** |
| full equalize | 79 | 140 | +4,304 | 113 | −4,341 | **−37** | **20,916** | **99.8%** |

校验：

- upsample 74：`20,182 + 3,645 + 771 = 24,598`
- upsample 79：`20,182 + 4,304 + 771 = 25,257`
- equalize 74：`255 × 74 + 771 = 18,870 + 771 = 19,641`
- equalize 79：`255 × 79 + 771 = 20,145 + 771 = 20,916`

`full equalize 79` 几乎不改变总行数（只少 37 行），但分布从“头重”变成“每画师 79 行”，头部的绝对曝光仍大幅下降。

### 3.2 step/epoch

`drop_last=True` 是**按 bucket** 丢尾，不是全局 `N // 8`：

$$
\text{steps/epoch} = \sum_{b \in buckets} \left\lfloor \frac{n_b}{8} \right\rfloor
$$

baseline 已由 `docs/bs8_5ep_launch.md` 验证为 **2,614**（11 个 bucket 求和），全局 floor 是 2,619，差 5。

本轮用本地图像尺寸按 `build_buckets(1048576)` + `assign_bucket` 复算，11 个 bucket 的计数与远端文档**逐桶完全一致**（838 / 1,235 / 974 / 692 / 1,941 / 1,273 / 1,086 / 5,721 / 4,208 / 2,119 / 866），因此下面的 per-bucket 估算是可信的；但最终仍以远端 `index.json` 脚本输出为准。**标记：估算，误差预期 ≤1–2 step/epoch。**

| 方案 | step/epoch（epoch 0–3 范围） | 4 epoch 总 step | 相对 10,456 |
|---|---:|---:|---:|
| baseline | 2,614 | 10,456 | — |
| upsample 74 | **3,069–3,070**（均值 3,069.75） | **12,279** | +1,823（+17.4%） |
| upsample 79 | **3,151–3,153**（均值 3,152.5） | **12,610** | +2,154（+20.6%） |
| equalize 74 | **2,450–2,452**（均值 2,450.75） | **9,803** | −653（−6.2%） |
| equalize 79 | **2,608–2,610**（均值 2,608.75） | **10,435** | −21（−0.2%） |

> epoch 间 ±1–2 的波动来自“余数图片轮换到不同 bucket”：总行数固定，但每桶余数变化会改变 `floor` 的和。4 epoch 总量稳定。
>
> 若不做轮换（每 epoch 固定同一批余数图片），每 epoch step 恒定，但会固定重复同几张图，不推荐。

### 3.3 固定旧 `max_steps` 时实际 epoch 数

| 方案 | 若 `max_steps=10,456` | 若 `max_steps=13,070` |
|---|---:|---:|
| upsample 74 | 10,456 / 3,069.75 ≈ **3.41 epoch** | ≈ 4.26 epoch |
| upsample 79 | ≈ **3.32 epoch** | ≈ 4.15 epoch |
| equalize 74 | ≈ **4.27 epoch** | ≈ 5.33 epoch |
| equalize 79 | ≈ **4.01 epoch** | ≈ 5.01 epoch |

结论：

- **upsample 74/79 必须改 `max_steps`**，否则“4 epoch”变成 3.3–3.4 epoch，尾部实际曝光 = 74×3.41≈252 次（仍高于 baseline 的 28×4=112 次，但不是设计目标）。
- **equalize 79 可以保持 10,456 附近**（4.01 epoch，10,435 更精确）。
- **equalize 74 会多出约 0.27 epoch 的预算**；若要保持 4 epoch，设 9,803；若要保持 10,456，则实际 4.27 epoch。
- 学习率 cosine 的 `total` 取 `max_steps`（`moss_lora/trainer.py:127-137`），改 step 数会同步改变衰减曲线；**不要只改数据不改 schedule**。

### 3.4 远端精确复算命令

在训练机（有 `/home/user/cache/clean30_v4_1m/index.json`）运行。脚本只依赖标准库，不需要 torch：

```bash
CACHE_INDEX=/home/user/cache/clean30_v4_1m/index.json
META=/mnt/data/train_test/clean30_top256/metadata.jsonl
TARGET=74          # 74 或 79
MODE=upsample      # upsample | equalize
SEED=42

python - "$CACHE_INDEX" "$META" "$TARGET" "$MODE" "$SEED" <<'PY'
import sys, json, hashlib, collections

index_path, meta_path, target, mode, seed = sys.argv[1:]
target = int(target); seed = int(seed)
meta = {}
for line in open(meta_path, encoding="utf-8"):
    row = json.loads(line)
    meta[row["image_sha256"]] = row["artist"]

index = json.load(open(index_path, encoding="utf-8"))
items = index["items"]
by_artist = collections.defaultdict(list)   # artist -> [(id, sha, bucket)]
for it in items:
    sha = it["image_sha256"]
    if sha not in meta:
        raise SystemExit(f"cache item {it['id']} sha not in metadata: {sha}")
    a = meta[sha]
    by_artist[a].append((it["id"], sha, (it["bucket_h"], it["bucket_w"])))

unknown = by_artist.pop("unknown")
counts = {a: len(v) for a, v in by_artist.items()}
assert len(counts) == 255, (len(counts), "named artists")
assert len(unknown) == 771, (len(unknown), "unknown rows")
print(f"items={len(items)} named_rows={sum(counts.values())} unknown={len(unknown)}")

def rank(a, sha):
    return hashlib.blake2b(f"{seed}:{a}:{sha}".encode(), digest_size=8).digest()

def counts_for_epoch(qmap, rmap, e):
    out = {}
    for a, rows in by_artist.items():
        c = len(rows); q = qmap[a]; r = rmap[a]
        ordered = sorted(rows, key=lambda x: rank(a, x[1]))
        start = (e * r) % c if c else 0
        extra = {ordered[(start + k) % c][1] for k in range(r)}
        for rid, sha, b in rows:
            out[rid] = (b, q + (1 if sha in extra else 0))
    for rid, sha, b in unknown:
        out[rid] = (b, 1)
    return out

qmap, rmap = {}, {}
for a, rows in by_artist.items():
    c = len(rows)
    if mode == "upsample" and c >= target:
        qmap[a], rmap[a] = 1, 0
    else:
        qmap[a], rmap[a] = divmod(target, c)

print(f"mode={mode} target={target} boosted={sum(1 for a,c in counts.items() if c < target)}")
for e in range(4):
    per = counts_for_epoch(qmap, rmap, e)
    bc = collections.Counter()
    for rid, (b, n) in per.items():
        bc[b] += n
    steps = sum(n // 8 for n in bc.values())
    print(f"epoch {e}: rows={sum(bc.values())} steps={steps} buckets={dict(sorted(bc.items()))}")
PY
```

预期（与本地复算一致，供对照）：

- `MODE=upsample TARGET=74`：rows=24,598，steps 约 3,070/3,070/3,070/3,069；
- `MODE=upsample TARGET=79`：rows=25,257，steps 3,153/3,153/3,153/3,151；
- `MODE=equalize TARGET=74`：rows=19,641，steps 约 2,452/2,450/2,451/2,450；
- `MODE=equalize TARGET=79`：rows=20,916，steps 2,608/2,610/2,608/2,609。

若远端输出与上述不同，以远端为准，并检查 `items` 是否恰好 20,953、`unknown` 是否 771。

---

## 4. 实现方案 A：sampler 级 repeat map（推荐）

### 4.1 为什么是 A

- 不复制图片、不重跑 VAE/text cache；
- 每个 epoch 可轮换余数图片；
- `drop_last`、bucket 分组、batch 形状、`_FixedBatches` resume 路径全部不变；
- 可精确控制“每个画师每 epoch 正好 T 张”；
- 唯一成本是 sampler 多一次展开（20k 项，毫秒级）。

### 4.2 数据依赖

cache 的 `index.json` 每项包含：

```json
{"id": "item_00001", "image": "...", "image_sha256": "...",
 "bucket_w": 1216, "bucket_h": 896, ...}
```

（`moss_lora/cache.py:596-599`、`moss_lora/cache.py:701-718`）

`metadata.jsonl` 每行包含 `image_sha256`、`artist`（`scripts/build_clean30_dataset.py:557-564`）。  
因此 **用 `image_sha256` 连接 metadata 和 cache 最稳**；`item_id` 会在重缓存后重排，sha 不会。repeat map 同时写 id 和 sha，加载时校验两者。

### 4.3 配置

在 `moss_lora/config.py:43-68` 的 `DataConfig` 增加：

```python
    # 画师均衡 repeat map（scripts/build_artist_repeat_map.py 生成）。
    # 空字符串 = 保持现状：每张图每 epoch 一次。
    repeat_map: str = ""
```

`__post_init__` 只做类型/空值检查；文件存在性在 trainer 启动时检查并给出明确错误。不要把 target/mode 放进训练配置，**target 固化在 map 里**，避免“配置说 74、map 是 79”的漂移。

YAML：

```yaml
data:
  repeat_map: /home/user/cache/clean30_v4_1m/artist_repeat_med74.json
```

### 4.4 repeat map 文件格式

```json
{
  "version": 1,
  "mode": "upsample",
  "target": 74,
  "seed": 42,
  "metadata_sha256": "<metadata.jsonl sha256>",
  "cache_index_sha256": "<index.json sha256>",
  "artist_counts": {"mignon": 298, "sy4": 28, "...": 0},
  "artists": {
    "sy4": {"count": 28, "target": 74, "base": 2, "extra": 18}
  },
  "items": {
    "item_01234": {"sha256": "<image sha>", "artist": "sy4",
                   "base": 2, "extra": 18, "rank": 7, "group": 28}
  },
  "expected": {"rows_per_epoch": 24598, "steps_per_epoch": [3070, 3070, 3070, 3069]}
}
```

- `base=q`，`extra=r`，`rank` 是哈希序下标，`group=c`；
- `unknown` 的项：`base=1, extra=0, rank=0, group=1`；
- `expected` 仅用于启动自检和打印，不参与采样；若与实际 cache 不一致则报错/警告。

### 4.5 生成器 `scripts/build_artist_repeat_map.py`（新增）

输入：

- `--metadata /mnt/data/train_test/clean30_top256/metadata.jsonl`
- `--cache-index /home/user/cache/clean30_v4_1m/index.json`
- `--mode upsample|equalize`
- `--target median|mean|<int>`（median→74，mean→79；也可显式 74/79）
- `--seed 42`
- `--out /home/user/cache/clean30_v4_1m/artist_repeat_med74.json`

步骤：

1. 读 metadata，建 `sha -> artist`；统计 `artist -> [item_id]`，**只从 cache items 统计**（保证训练口径）；
2. 校验：255 个有名画师、`unknown=771`、`len(items)=20,953`、每个 item 的 sha 在 metadata 中；
3. 目标：`median` 用 `statistics.median`（255 个值 → 74），`mean` 用 `round(mean)`（79）；
4. 对每个画师：按 `blake2b(f"{seed}:{artist}:{sha}")` 排序；`q,r=divmod(T,c)`；`upsample` 模式下 `c>=T` 强制 `q=1,r=0`；`rank=k`；
5. 生成 `items` / `artists` / `expected`，写 JSON（`sort_keys=True`，UTF-8，原子替换）；
6. 打印自检：每画师 `c*q+r == T`（或头部 1×c）、总行数、每 epoch 的 step（按 cache bucket 直接求和）。

运行：

```bash
python scripts/build_artist_repeat_map.py \
  --metadata /mnt/data/train_test/clean30_top256/metadata.jsonl \
  --cache-index /home/user/cache/clean30_v4_1m/index.json \
  --mode upsample --target median --seed 42 \
  --out /home/user/cache/clean30_v4_1m/artist_repeat_med74.json
```

### 4.6 Sampler 改动（`moss_lora/dataset.py:114-162`）

当前 `BucketBatchSampler`：

- `__init__` 在 120–130 行按 bucket 分组；
- `set_epoch` 132–133；
- `batches_for_epoch` 135–149 复制、shuffle、按 `batch_size` 切、`drop_last`、再 shuffle；
- `__len__` 154–162 用 `len(indices)//batch_size` 求和。

建议改动：

```python
class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, buckets, batch_size, drop_last=True, seed=42, epoch=0,
                 repeats: list[tuple[int, int, int, int]] | None = None):
        ...
        self.repeats = repeats          # 每项 (base, extra, rank, group)
        if repeats is not None and len(repeats) != len(buckets):
            raise ValueError("repeat map length != dataset length")
        self._cached_epoch = None
        self._cached_batches = None

    def _repeat_count(self, index: int, epoch: int) -> int:
        if self.repeats is None:
            return 1
        base, extra, rank, group = self.repeats[index]
        if extra and (rank - epoch * extra) % group < extra:
            return base + 1
        return base

    def batches_for_epoch(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        if self._cached_epoch == epoch:
            return self._cached_batches
        rng = random.Random(self.seed + epoch)
        batches = []
        for indices in self.groups.values():
            expanded = []
            for i in indices:
                expanded.extend([i] * self._repeat_count(i, epoch))
            rng.shuffle(expanded)
            if self.drop_last:
                batches.extend(expanded[i:i+self.batch_size]
                               for i in range(0, len(expanded)-self.batch_size+1, self.batch_size))
            else:
                batches.extend(expanded[i:i+self.batch_size]
                               for i in range(0, len(expanded), self.batch_size))
        rng.shuffle(batches)
        self._cached_epoch, self._cached_batches = epoch, batches
        return batches

    def __len__(self):
        if self.repeats is None:
            # 保留原有 O(bucket) 公式，兼容现有测试
            ...
        return len(self.batches_for_epoch(self.epoch))
```

要点：

- **展开后再 shuffle**，而不是“先 shuffle 再重复”，否则同一图片的副本会相邻；当前实现是每桶 `rng.shuffle(indices)` 后按顺序切，展开后再 shuffle 更均匀。
- `drop_last` 在展开后的桶内做，语义与现有完全一致。
- `__len__` 现在依赖 epoch；`trainer.py:446-447` 已经先 `set_epoch(epoch)` 再 `batches_for_epoch(epoch)`，所以一致。
- `trainer.py:527` 的 `len(sampler)` 在循环退出时使用当前 epoch；若 `micro_in_epoch=0`（epoch 边界）比较结果不受 ±1 影响；若中途停止，`micro_in_epoch` 是当前 epoch 已消费的微批数，`len` 也是当前 epoch，语义正确。
- 若不想改类，可写 `ArtistBalancedBucketSampler(BucketBatchSampler)` 覆写 `__init__/batches_for_epoch/__len__`；但 trainer 里仍要改构造调用，收益不大。**推荐直接扩展原类**，只保留一条路径。

### 4.7 Trainer 改动

1. **启动加载**（`moss_lora/trainer.py:410-413`）：

```python
    dataset = CachedDataset(config.data.cache_dir,
                            variant_weights=config.data.resolved_variant_weights() or None)
    repeats = load_artist_repeat_map(config.data.repeat_map, dataset.items) \
        if config.data.repeat_map else None
    sampler = BucketBatchSampler(..., repeats=repeats)
    if repeats is not None:
        print(f"[data] repeat map on: {sum(...)} views/epoch, {len(sampler)} steps/epoch")
```

`load_artist_repeat_map` 放 `dataset.py`，职责：

- 读 JSON，校验 `version`；
- 对每个 `dataset.items`：id 必须存在，sha 必须匹配；
- `artist_counts` 必须与 cache 实际计数一致；
- 返回 `[(base, extra, rank, group)]`，`unknown` 为 `(1,0,0,1)`；
- 任何不一致直接 `ValueError`，不要静默降级。

1. **epoch 循环**（`trainer.py:445-450`）：不需要改，`set_epoch` + `batches_for_epoch` + `batches[micro_in_epoch:]` 已兼容。
2. **checkpoint 记录**（`trainer.py:260-277`）：

```python
"repeat_map_sha256": _repeat_map_digest(config.data.repeat_map),
"repeat_map_target": ...,
"repeat_map_mode": ...,
```

`_repeat_map_digest` 用 `moss_lora.utils.sha256_file`（已有，`utils.py:28`）；空路径返回 `None`。
3. **resume 校验**（`trainer.py:292-314`）：读取 `payload["repeat_map_sha256"]`，与当前文件 digest 比较；不一致直接报错：

```
RuntimeError: repeat map changed since checkpoint (stored=..., current=...);
resume would renumber batches. Restore the original map or start a new run.
```

原因：`batch_index` 只是“当前 epoch 已消费的微批数”，它只在同一 `(seed, epoch, repeat map)` 下对应同一 `batches` 列表。map 的 target/mode/rotation 任一变化都会让 `batches[batch_index:]` 指向不同数据，必须拒绝。

4. **不要在 resume 时重新计算 target**：target 固化在 map 中；配置只存路径。

### 4.8 测试

- `tests/test_cache_dataset.py:414-432` 现有 `test_dataset_and_sampler` 继续跑（`repeats=None` 路径）。
- 新增：
  - 固定 4 个 item、2 个 bucket、`T=3`：每 epoch 每个画师恰好 3 个 view；`len(sampler)==len(sampler.batches_for_epoch(e))`；
  - 两个 epoch 的额外图片集合不同（轮换），但每个画师每 epoch 总数相同；
  - `set_epoch` 后 `batches_for_epoch(e)` 与直接调用一致；
  - 非法 map（长度不符、sha 不符、artist count 不符）报错；
  - `tests/test_trainer_smoke.py:142-167` 加一条“带 map 的 resume 与不中断的 batch id 序列完全一致”。
- `tests/test_trainer_smoke.py:170-200` 的 epoch-end 游标测试应在 `repeats=None` 和 `repeats` 两种情况下都通过；epoch-end 仍写 `(epoch+1, 0)`（`trainer.py:519-523`），与采样器无关。

### 4.9 优点/代价

| 项 | 评价 |
|---|---|
| 数据/缓存 | 不动，零重复图片成本 |
| 代码量 | 约 80–120 行（sampler + loader + config + digest） |
| 精确性 | 每画师每 epoch 精确 T；可复现 |
| resume | 只要 map + seed 不变，精确；否则显式拒绝 |
| 风险 | 采样器展开多一次内存（24k 索引，可忽略）；需要新脚本生成 map |

---

## 5. 实现方案 B：平衡 JSONL + 重新缓存（快速 pilot）

### 5.1 做法

不要用文件夹模式：`_collect_folder` 每个图片文件只收一次（`moss_lora/cache.py:448-463`），无法通过重复文件表达重复。必须用 JSONL：

1. 读 `metadata.jsonl` 的 20,953 行，**原顺序**写入 `balanced_med74.jsonl`；
2. 对每个 `c<T` 的画师，计算 `q,r`；
   - 所有该画师行再追加 `q-1` 份；
   - 再选 `r` 行各追加 1 份（固定按 sha 哈希序，不轮换）；
3. 头部行不追加；`unknown` 不追加；
4. 用该 JSONL 重新 `moss-lora-cache`。

关键行数：24,598（T=74）或 25,257（T=79）。

### 5.2 注意点

- **JSONL 的 `file_name` 是相对于 JSONL 所在目录**（`cache.py:439-452`）。最稳妥：把 `balanced_med74.jsonl` 放在 `/mnt/data/train_test/clean30_top256/`，与 `images/` 同级；否则把每行改成绝对 `image` 字段。
- 重复行会产生新的 `item_XXXXX` id；因为原 20,953 行在前、cache 按行顺序生成 id，**前 20,953 个 id 与原 cache 一致**，`logging.sample_val_ids`（`item_00000` 等）仍然有效。
- cache 体积按行数比例增长：T=74 **+17.4%**，T=79 **+20.6%**；latent + text 全部重复存储。若原 cache 是 100 GB，则分别约 +17.4 GB / +20.6 GB。
- 每个 epoch 都是同一批重复项，**没有余数轮换**；28 张画师的那 18 张“额外图”永远相同，4 epoch 后重复模式固定。作为 pilot 可接受，正式实验不建议。
- 全均衡 pilot 不可取：必须删除头部行，原 id 顺序被打乱，`sample_val_ids` 失效；且 cache 要重编码。若一定要做，构建后从新 `index.json` 按 image sha 反查 id，重写 `sample_val_ids`。
- 缓存命令（示例）：

```bash
moss-lora-cache \
  --input /mnt/data/train_test/clean30_top256/balanced_med74.jsonl \
  --output /home/user/cache/clean30_v4_1m_bal74 \
  --model /home/user/models/MOSS-Image-Base \
  --target-area 1048576 \
  --max-text-length 1024 \
  --negative-prompt "worst quality, bad anatomy, watermark, jpeg artifacts, bad hands, extra fingers, missing fingers"
```

- 建议新目录，不要覆盖原 cache；否则 `cache.py:720` 会清理索引外旧文件，且无法回滚 A/B。
- 优点：零代码改动、最快验证“补尾部是否有效”；缺点：重缓存时间 +17–21%、无轮换、磁盘增加、A/B 需要两份 cache。

### 5.3 Pilot vs 正式跑建议

| 目的 | 选择 |
|---|---|
| 先验证“补尾部是否改善 tail 风格、是否伤 head” | **方案 B，T=74，upsample-only**；约 24.6k 行，重缓存 +17.4%，跑 4 epoch（~12,280 step） |
| 正式实验、需要轮换/精确 resume/A-B 复用同一 cache | **方案 A，repeat map** |
| 探索全均衡 | 只在 A 上做；B 的全均衡需要重排 val id，不值得 |

---

## 6. 实现方案 C：WeightedRandomSampler（不推荐）

`torch.utils.data.WeightedRandomSampler(..., replacement=True)` 的问题：

1. **与 bucket batching 冲突**：`collate_batch` 明确要求一个 batch 内所有 latent 形状相同（`dataset.py:182-184`），随机采样会把不同 `(h,w)` 混进同一 batch，直接 `RuntimeError`。只能每个 bucket 单独 weighted sampler，再拼接。
2. **每 epoch 的 batch 数随机**：每个 bucket 抽多少是随机的，`sum(//8)` 不再是固定值；`max_steps`、cosine schedule、epoch 边界全部失去确定含义。`drop_last` 也变成随机丢尾。
3. **resume 复杂**：需要保存 sampler 的 RNG 状态 + 每个 DataLoader worker 的 RNG；现有 resume 依赖 `_FixedBatches` + `batch_index` 精确快进（`dataset.py:164-175`、`trainer.py:448-449`），weighted sampler 无法给出“第 k 个微批”的稳定映射。
4. **曝光不可控**：有放回采样下某张图在一个 epoch 内可能 0 次或 5 次；无法保证“每个画师正好 74 张”，方差大。
5. **同 batch 重复**：同一张图可能被抽进同一个 batch，浪费 batch 多样性。

若一定要用，只能“按 bucket + 固定每桶采样数 + 保存 RNG”，复杂度高于 A，收益没有。**不推荐。**

---

## 7. 风险与缓解

### 7.1 记忆化（最大风险）

`T=74` 时：

- 47 个 28–36 张的画师，每 epoch 每张 2–3 次；4 epoch 后每张 **8–12 次**；
- 头部（upsample-only）每张 4 次；
- 28 张画师平均每张 10.6 次，是头部的 2.6 倍。

当前 cache 是**固定中心裁剪 latent**，训练循环没有 on-the-fly 几何/颜色增强；重复同一 latent 会强化构图记忆。caption 有 5 个变体（`cache.py:60-68`），每次 view 独立抽 variant（`dataset.py:51-62`），文字侧有变化，但图像侧完全一致。

缓解：

- 先做 **T=74 upsample-only**，不要一步到 79 或全均衡；
- 对 28–36 张画师做**质量/近重复过滤**：同一画师内先做感知哈希/CLIP 近重复聚类，删掉重复构图再补；
- 设 **max repeat cap**：若担心 3×，把 28–36 张画师的 3× 压到 2×。代价约 **500 行/epoch**（24,098 vs 24,598，估算），step 约少 62；先跑 cap=3 的 pilot，看记忆化指标再决定；
- 监控：生成图对同画师训练图的最近邻 cosine、生成多样性（pairwise LPIPS/CLIP）、tail 画师的 prompt 变化下构图是否单一；
- 不要在同一个 run 里同时改 `variant_weights`、LR、epoch 数；否则无法归因。

### 7.2 头部仍然占优（upsample-only）

T=74 upsample 后的分层占比：

| 分层 | 行数 | 占比 |
|---|---:|---:|
| tail | 9,398 | 38.2% |
| mid | 11,080 | 45.0% |
| head | 3,349 | 13.6% |
| unknown | 771 | 3.1% |

tail 从 27.5% 提到 38.2%，但 head 仍有 13.6%、mid 45.0%。这是**有意的**：头部是风格质量的主要来源，不能为了“平均”把 mignon/sincos 这类高价值画师压掉。若目标是“每个画师同等重要”而不是“尾部不被忽略”，才考虑全均衡。

### 7.3 全均衡会丢头部数据

`equalize 74` 每 epoch 删 4,957 行头部样本（127 个 >74 的画师），头部从 3,349 行降到 1,332 行（6.8%）；mignon 从 298 行/epoch 降到 74 行/epoch。

轮换的补偿：4 epoch 内 mignon 的 74 张/epoch 轮换覆盖 296/298 张，**唯一图片覆盖率接近 100%**，但每张只出现约 1 次（vs baseline 4 次）。所以不是“丢图片”，而是“丢重复曝光”。对于风格 LoRA，头部风格的**每图梯度次数**从 4 降到 1，可能欠拟合；不建议作为第一实验。

### 7.4 其他

- **unknown 771 行**不参与均衡；upsample 后占比从 3.7% 降到 3.1%，影响很小。不要把 `unknown` 当画师去补。
- **多画师行**：当前 metadata 的 `artists` 全为空、`artist` 单一，公式按主画师计一次。未来若出现多画师行，必须先定义主画师，否则同一行会被多个画师重复计数。
- **val 泄漏**：`logging.sample_val_ids` 是训练集内的 item（`configs/train_5090_v4_5ep_r64_bs8.yaml:44-45`），不是 held-out。补样不会改变这些 id 的语义（方案 A/B 都保留原 id），但**不要把补样图当成验证集**。
- **resume 漂移**：改 target/mode/rotation 后从旧 checkpoint resume，`batch_index` 不再对应同一批数据；必须用 digest 拒绝（见 4.7）。
- **磁盘/显存**：方案 A 不增加 cache；方案 B 增加 17–21% 磁盘。显存不变（batch 仍 8，形状不变）。
- **step 统计口径**：`metrics.csv` 的 `step` 是 optimizer step；`gradient_accumulation_steps=1`（配置），所以 step = micro-batch。若以后改 accum，`batches/epoch` 与 `steps/epoch` 要重新换算。

---

## 8. 与 caption 计划的交互

### 8.1 变体权重是“绝对权重”，不是“相对比例”

`data.variant_weights` 在 `config.py:49-54`，`resolved_variant_weights()` 在 `config.py:70-85`，`pick_variant` 在 `dataset.py:51-62`：

- 传入 `{"short": 0.25}` 时，**未列出的 kind 权重为 0**；
- 若某个 item 没有 `short`，该 item 的所有 kind 权重和为 0，会 fallback 到**均匀分布**（`dataset.py:58-60`）；
- 因此要降 tag-only 权重，必须**显式写全五个 kind**，例如：

```yaml
data:
  variant_weights:
    original: 0.16
    long: 0.24
    medium: 0.24
    short: 0.20
    layout: 0.16
```

（和为 1.0 只是便于阅读；`random.choices` 只看向对权重，不要求和为 1。）

### 8.2 画师标签曝光怎么变（事实核查）

在 clean30 中，**五个缓存变体全部带 `artist:<id>`**，包括 `original`。`caption_original` 不是“只含 tag”的裸文本，而是带完整前缀/prose 的源 caption，例如：

```text
@CH3OCH3, artist:okkimon, aru (blue archive), 4girls, ... , <prose>
```

`long/medium/short` 也都是 `<trigger>, artist:<id>, character:<id...>, <tags>, <prose 或截断>`（`docs/CAPTIONS.md:126-137`）。因此：

$$
\text{每 epoch 画师标签 view 数} = T
$$

不是 `T × (1 - w_original)`。默认权重下也不存在“约 84% 的 view 带标签”这回事——**每一个 view 都带标签**。

- `T=74` → 每 epoch 74 个带标签 view/画师；baseline 28 张 → 28 个。**尾部画师标签曝光 74/28 = 2.64×**；4 epoch 总计 296 vs 112。
- 把 `short` 从 0.213 降到 0.20–0.30：**artist tag 出现率完全不变**（short 本身带标签），只是把“短 tag vs prose”的 token 监督比例挪动几个百分点。
- 把 `original` 权重调高也不会降低 artist tag 曝光；`original` 同样带标签。它改变的是“原始 caption 风格 vs 重建变体风格”的文本分布。
- 如果新 caption 计划是“剥掉 long/medium 里的通用 style prose，只留 artist/character/tags”，那么 artist tag **仍在**；只有引入一个**不含 artist 的新变体**、或做 artist dropout，才会降低标签曝光。届时才需要按 `Σ_{k 含标签} w_k` 重算。
- “strip generic style prose” 减少的是通用风格描述的 token 先验，不改变采样次数；它会让模型更依赖 `artist:` 前缀，因此**均衡采样 + 去 prose 同时做会放大画师条件**，需要 A/B 验证是否过拟合。

> 结论：不要把 `original` 当作“无 artist tag”的变体；在 clean30 上，均衡采样对 artist tag 的曝光就是 `T`，与 variant 权重无关（只要不引入无标签变体）。

### 8.3 step/epoch 预算

- upsample 74：3,069.75 step/epoch；4 epoch = **12,279**。若仍用 10,456，则只有 3.41 epoch。
- upsample 79：3,152.5；4 epoch = **12,610**。
- equalize 74：2,450.75；4 epoch = **9,803**；若想保持 4 epoch 预算不变，可跑 4.27 epoch（10,456）或 5 epoch（12,254）。
- equalize 79：2,608.75；4 epoch = **10,435**，几乎等于现有 10,456。

建议：

- **保持 4 个完整 epoch**，把 `max_steps` 改成对应值（74→12,279；79→12,610；eq74→9,803；eq79→10,435）；
- `warmup_steps=200` 不变；cosine 的 `total` 自动跟随 `max_steps`（`trainer.py:127-137`）；
- `save_every=3000`、`sample_every=3000` 不变，但注意 epoch 边界不再是 3,000 的整数倍；epoch-end checkpoint 由 `save_epoch: true` 负责（`trainer.py:519-523`）；
- 如果为了省 GPU 坚持 10,456：upsample 74 仍有 3.41 epoch，尾部曝光 74×3.41≈252 次，比 baseline 112 次好，但“4 epoch”说法不成立；报告里必须写实际 epoch 数。

固定 step 预算的稀释效应（以 upsample 74 为例，估算）：

| 设置 | 实际 epoch | 头部单图曝光（c=298） | 28 张画师总曝光 | 说明 |
|---|---:|---:|---:|---|
| baseline, 4 epoch | 4.00 | 1,192 | 112 | 基准 |
| upsample 74, 固定 10,456 step | 3.41 | 1,015（×0.85） | 252（×2.25） | 头/中段被稀释，尾部相对提升变小 |
| upsample 74, 4 epoch（12,279 step） | 4.00 | 1,192（×1.00） | 296（×2.64） | 设计目标 |

**建议：正式跑用 4 个完整 epoch（12,279）；A/B 主比较用固定总 step 12,280（compute-matched），两种口径都在实验记录里写清楚。**

---

## 9. A/B 实验设计

### 9.1 三条臂

| 臂 | 数据 | 每 epoch step 估算 | 说明 |
|---|---|---:|---|
| A0 | baseline，无 repeat map | 2,614 | 现有 recipe |
| A1 | upsample-only，T=74 | 3,070 | **首选** |
| A2 | full equalize，T=74 | 2,451 | 风险对照 |

可选 A2b：equalize 79（2,609），用于“几乎不增加行数”的均衡对照；但首轮不要铺太多臂。

### 9.2 固定预算与 seed

- **主比较固定总 optimizer step = 12,280**：
  - A0：12,280 / 2,614 = **4.70 epoch**；
  - A1：**4.00 epoch**；
  - A2：12,280 / 2,451 = **5.01 epoch**。
  - 这样三臂计算量相同，cosine schedule 总步数相同，差别只在数据分布。A0 的 4.70 epoch 是“同样的 step 预算下多看 0.7 epoch”，这是公平的 compute-matched 对照。
- **次比较固定每臂 4 epoch**：A0 10,456、A1 12,279、A2 9,803；用于回答“4 epoch 配方哪个好”，但计算量不同。
- **caption 权重在所有臂中完全一致**：`data.variant_weights`（包括 `short` 的 0.2–0.3 调整）和 prose 处理必须作为独立变量，不能和均衡采样同轮变化；否则无法归因。先固定当前 caption 配方，跑完 A/B 再单独评估 caption 计划。
- 训练 seed：`train.seed=42`；正式结论用 **42/43/44** 三个 seed。每个 seed 只改 sampler/timestep RNG，数据顺序和 prompt 固定。
- 采样 seed：固定 `1234`；每个 checkpoint 用同一组 latent noise 和 prompt，避免生成随机性掩盖差异。
- 采样参数与现配置一致：30 step、CFG 3.0、`sample_skip_layer=22`、同一 negative prompt（`configs/train_5090_v4_5ep_r64_bs8.yaml:40-43`）。

### 9.3 分层指标

按 **head (≥150, n=18) / mid (74–149, n=110) / tail (<74, n=127)** 分层报告；`unknown` 单列。

| 指标 | 定义 | 目的 |
|---|---|---|
| Artist top-1 | 用固定 prompt（不带 artist tag）生成；用图像 encoder（DINOv2/CLIP）对**每画师等量参考图**（建议 8 张/画师）做最近邻，是否命中目标画师 | 画师可辨识性 |
| Artist margin | `sim(top1) - sim(top2)` 的均值 | 置信度，不只看是否第一 |
| Style fidelity | 生成图与目标画师参考集 centroid 的 cosine；另做 50 prompt/band 的人工 pairwise | 风格保真 |
| Memorization | 生成图对**同画师训练图**的最大 cosine；以及同一 prompt 多 seed 的 pairwise 多样性 | 是否复刻训练图 |
| Prompt adherence | 对非画师 tag（角色/服装/背景）用 tagger 计算 recall/precision | 是否因补样伤害通用条件 |
| FID/KID | 若有 held-out 集，按 band 计算 | 分布质量 |
| Loss | 若有 held-out，按 band 的 val loss；否则只看 train loss，不能当泛化 | 辅助 |

参考集注意：

- 每画师参考图数量必须相同（8 张），否则 head 的类内方差更大，top-1 会被动偏高；
- 若从训练集抽参考图，属于泄漏，只能做**相对比较**，不能报绝对 top-1；
- 更干净的做法：训练前每画师留出 5–8 张不参与 cache，作为参考/val。这样训练行数减少（255×5≈1,275 行），目标计数和 step 要重算；A/B 三臂使用同一 held-out 划分。

### 9.4 评估时点

- 固定 step：3,000 / 6,000 / 9,000 / 12,280；若 4-epoch 次比较，再加 epoch-end。
- 每个时点生成：每画师 5 个固定 prompt（共 1,275 张/臂/时点），固定 seed；人工 pairwise 只抽 50 prompt/band。
- 报告：均值 ± bootstrap 95% CI（按画师重采样，不按图片，避免 head 图片多主导）。

### 9.5 判定规则（建议写进实验记录）

- **A1 胜出**：tail top-1 +≥3 个百分点，head top-1 下降 ≤1 个百分点，memorization NN cosine 上升 ≤0.02，mid 不掉。
- **A2 可接受**：tail +≥5 个百分点，head 下降 ≤2 个百分点，且 memorization 不显著上升；否则不采用全均衡。
- **A0 保持**：A1 的 tail 提升 <1 个百分点，或 head 下降 >2 个百分点。
- 任何臂出现 tail 生成构图单一、同画师 NN cosine 显著上升（>0.05）→ 先降 max repeat cap 或加质量过滤，不要加 epoch。

---

## 10. 最终建议（按优先级）

1. **第一枪：方案 A（repeat map）+ upsample-only T=74 + 4 epoch（max_steps=12,279）**。不删头部、不重缓存、可轮换、可 resume；先回答“补尾部是否有用”。
2. **pilot 可选方案 B**：如果不想改代码，用 `balanced_med74.jsonl` + 新 cache 跑 4 epoch；但必须接受固定重复、+17.4% 磁盘/缓存时间，且不能做全均衡。
3. **不要第一枪做全均衡 74**：删 4,957 行头部/epoch，风格保真风险最大；等 A1 证明尾部收益后，再用 A2 做“等 compute 12,280 step（5.01 epoch）”对照。
4. **不要用 WeightedRandomSampler**：与 exact-shape bucket batch、固定 step、精确 resume 三条都冲突。
5. **caption 权重改动与采样改动分开 A/B**；`data.variant_weights` 要写全五个 kind，避免“只写 short 导致其他 kind 权重 0 / 缺失 kind 回退均匀”的陷阱。

---

## 11. 附：关键文件与行号索引

| 内容 | 位置 |
|---|---|
| DataConfig / variant weights | `moss_lora/config.py:43-85` |
| CachedDataset / index 读取 | `moss_lora/dataset.py:66-113` |
| BucketBatchSampler | `moss_lora/dataset.py:114-162` |
| `set_epoch` | `moss_lora/dataset.py:132-133` |
| `batches_for_epoch` | `moss_lora/dataset.py:135-149` |
| `__len__` | `moss_lora/dataset.py:154-162` |
| `_FixedBatches` | `moss_lora/dataset.py:164-175` |
| `collate_batch` 形状校验 | `moss_lora/dataset.py:182-184` |
| trainer 构造 sampler | `moss_lora/trainer.py:410-413` |
| epoch 循环 / resume 切片 | `moss_lora/trainer.py:445-450` |
| checkpoint 状态 | `moss_lora/trainer.py:260-277` |
| resume 读取 | `moss_lora/trainer.py:292-314` |
| 最终 `len(sampler)` | `moss_lora/trainer.py:527-531` |
| cosine schedule 依赖 max_steps | `moss_lora/trainer.py:127-137` |
| cache 记录 / bucket | `moss_lora/cache.py:165-167`, `moss_lora/cache.py:596-599` |
| cache index 输出 | `moss_lora/cache.py:701-718` |
| metadata 构建 | `scripts/build_clean30_dataset.py:557-564` |
| 现有 sampler 测试 | `tests/test_cache_dataset.py:414-432` |
| resume 测试 | `tests/test_trainer_smoke.py:142-167`, `170-200` |
| 当前 bs8 配置 | `configs/train_5090_v4_5ep_r64_bs8.yaml` |
| 现有 step 数学 | `docs/bs8_5ep_launch.md:10-32` |
