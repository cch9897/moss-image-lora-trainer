# cleanV2 implementation

Status: generated and validated on both local inputs and the remote WSL corpus.
The deployed remote manifest is the **tightened** conservative strip (§4.1);
the cache build is running from it and training is **not** launched yet. The
pre-tighten manifest is retained as `metadata_before_tighten.jsonl` for
rollback.

- Generator: `scripts/make_clean_v2.py`
- Remote deploy helper: `scripts/deploy_cleanv2_remote.sh`
- Remote manifest: `/home/user/dataset/cleanV2/metadata.jsonl`
- Remote cache (in progress): `/home/user/cache/cleanV2_1m`
- Old cache kept for rollback: `/home/user/cache/clean30_v4_1m` (untouched)

## 1. What cleanV2 changes

`cleanV2` is a balanced manifest plus a rebuilt cache. No sampler/training code
changes.

1. **True tag-only `short`**: `prefix + tags[:12]`, no prose. The v4 finalize
   path put a short prose sentence into `caption.short`; cleanV2 drops it.
2. **Conservative prose strip**: clause-level removal of generic style/medium/
   lighting/palette/quality prose from `caption.long`, `caption.medium`,
   `caption_original`, `caption_no_artist`, `layout.summary.caption`, and
   prose-like layout item `caption` fields. Layout item `text` fields are never
   touched. `caption_no_artist` is never reduced to tag-only: if stripping would
   empty it, the original prose is kept.
3. **Upsample-only artist balance**: target 74 (median of the 255 known artists).
   For `c < 74`, `q, r = divmod(74, c)`; each of the `c` images gets `q` copies
   and the first `r` images get one extra, so the artist contributes exactly 74
   rows per epoch. Head artists keep every row; unknown/no-artist rows stay at
   1x. Repeats are appended after all originals, so the first 20,953 cache item
   IDs and existing `sample_val_ids` remain valid.

## 2. Usage

```bash
python scripts/make_clean_v2.py \
  --metadata /path/to/metadata.jsonl \
  --output /path/to/cleanV2 \
  [--images-root /path/to/dataset] \
  [--limit N] [--target 74] [--examples 8]
```

Defaults:

- `--images-root`: the metadata file's parent directory.
- `--target`: rounded median of known-artist base counts.
- `--trigger`: auto-detected from the first caption (`@CH3OCH3` in this corpus).
- Validation is strict: any hard failure exits non-zero unless `--no-strict`.

Every output row keeps all original fields and gains:

- `dataset_version: "cleanV2"` on every row;
- absolute `image` (prefers an existing `source_path`, else resolves
  `file_name` against `--images-root`), so the JSONL can live in
  `/home/user/dataset/cleanV2/` without symlinks;
- on repeats only: `repeat_of` (original `file_name`) and 1-based
  `repeat_index`.

Outputs: `metadata.jsonl`, `report.json`, `README.md`.

## 3. Exact counts

### Local full clean30 row set (`/mnt/data/train_test/clean30_top256/metadata.jsonl`)

| Metric | Value |
|---|---:|
| Base rows | **20,953** |
| Balanced rows | **24,598** |
| Appended repeats | **3,645** |
| Known artists / unknown rows | **255 / 771** |
| Artist target (median) | **74** |
| Artists boosted | **127** |
| Tag-only `short` (base) | **20,953** |
| Tag-only `short` (balanced) | **24,598** |
| Empty captions | **0** |
| Missing images | **0** |
| Quoted spans lost | **0** |

Copies per image: 2,608 images at 1 copy, 2,645 at 2 copies, 500 at 3 copies.
Repeat-index histogram: 3,145 second copies, 500 third copies. Every boosted
artist has exactly 74 rows; head artists keep their full count (max 298).

### Remote full v4 (`/home/user/dataset/clean30_top256/final_v4/metadata.jsonl`)

Same row/artist numbers: 20,953 base, 24,598 balanced, 3,645 repeats, 255
known artists + 771 unknown, target 74, 127 boosted, tag-only short 20,953
base / 24,598 balanced, 0 empty captions, 0 missing images, validation PASS,
0 quoted spans lost.

### Local 1k v4 prose smoke test (`/mnt/data/train_test/ab_1k/final/metadata.jsonl`)

1,000 base rows, median 3, 1,165 balanced, 165 repeats, 108 artists boosted,
tag-only short 1,000 base / 1,165 balanced, 0 empty captions, validation PASS.

## 4. Style strip

The filter is deliberately conservative. It removes only:

- leading media/medium frames such as
  `A digital anime-style illustration of a girl ...` -> `A girl ...`;
- whole sentences that are pure style/mood boilerplate;
- style-only comma clauses (`soft lighting`, `clean lines and soft shading`,
  `rendered in a soft, painterly style`, `An image, ...` replacements);
- quality boilerplate (`highly detailed`, `best quality`, `8k`, ...).

It never globally deletes `anime`, `detailed`, `monochrome`, `sketch`,
`watercolor`, `pastel`, or `anime figure`. A clause is removable only when all
of its tokens are style/stop words and it contains a strong style token.
Quoted text (double, curly, and single-quoted phrases), NSFW terms, visible text,
concrete colour enumerations, and content nouns block removal. Finite-verb
clauses are not removed (avoids `The lighting is soft, reflecting ...` ->
fragment), and a clause whose preceding clause ends in an adjective/preposition
is kept (avoids `with a soft, glowing quality` -> `with a soft.`).

### Strip rates

Local full clean30:

| field | changed / fields | rate | chars removed |
|---|---:|---:|---:|
| `caption.long` | 4,439 / 20,953 | 21.2% | 181,058 |
| `caption.medium` | 4,467 / 20,953 | 21.3% | 181,686 |
| `caption_original` | 4,356 / 20,953 | 20.8% | 178,551 |
| `caption_no_artist` | 4,356 / 20,953 | 20.8% | 178,551 |
| `layout.summary.caption` | 2,056 / 15,114 | 13.6% | 79,540 |
| layout item `caption` | 0 / 0 | 0% | 0 |

Local 1k v4:

| field | changed / fields | rate | chars removed |
|---|---:|---:|---:|
| `caption.long` | 98 / 1,000 | 9.8% | 3,115 |
| `caption.medium` | 132 / 1,000 | 13.2% | 3,932 |
| `caption_original` | 0 / 1,000 | 0% (v4 original is tag-only) | 0 |
| `caption_no_artist` | 98 / 1,000 | 9.8% | 3,105 |
| `layout.summary.caption` | 213 / 810 | 26.3% | 3,759 |
| layout item `caption` | 0 / 94 | 0% | 0 |

Remote full v4:

| field | changed / fields | rate | chars removed |
|---|---:|---:|---:|
| `caption.long` | 5,420 / 20,953 | 25.9% | 251,838 |
| `caption.medium` | 3,000 / 20,953 | 14.3% | 92,259 |
| `caption_original` | 4,361 / 20,953 | 20.8% | 178,901 |
| `caption_no_artist` | 5,420 / 20,953 | 25.9% | 251,838 |
| `layout.summary.caption` | 1,356 / 13,603 | 10.0% | 39,435 |
| layout item `caption` | 0 / 0 | 0% | 0 |

Remote v4 dropped **2,115,536** characters of short prose by replacing
`caption.short` with tag-only text. Quote-guard fallbacks (field left unchanged
to preserve quoted text) occurred on 619 long / 207 medium / 204 layout /
5 original fields; validation confirmed 0 quoted spans were lost.

### 4.1 Strip tightening (adopted)

The independent audit found clear generic boilerplate surviving the first
strip. The generator was patched conservatively:

- `LEADING_MEDIUM` adjectives extended with `vibrant|bright|colorful|
  saturated|minimalist|soft|warm|cool|elegant|stylized|intricate|
  highly detailed` (plus colourless spellings), and `one girl` -> `a girl`
  normalization after the frame is removed.
- New whole-sentence `_STYLE_SUBJECT_SENTENCE` rule for
  `The art style features/is/uses/showcases ...`, `The style ...`, and
  `The rendering/lighting/palette is ...`. A sentence is removed only when its
  predicate contains only style/stop vocabulary, with no content noun, no NSFW
  term, no quoted text, and no finite content verb; content sentences such as
  `Soft warm dusk lighting illuminates the dark wooden shoji doors.` are kept.
- Seven exact regression cases (three strip, four keep) run on every
  generation; their results are written to `report.json` `regressions`.
  `--examples N` now records before/after prose pairs in `report.json`.

Remote result (base 20,953 rows, prose fields only, tag sections excluded):

| term | before | after | cut |
|---|---:|---:|---:|
| `anime-style` | 1,197 | 1,170 | 2.3% |
| `cel shading` | 1,024 | 790 | 22.9% |
| `soft shading` | 1,490 | 1,182 | 20.7% |
| `painterly` | 609 | 558 | 8.4% |

Safety: 0 NSFW terms lost, 0 quoted spans lost, 0 layout item `text` changes,
0 empty captions, validation PASS, 7/7 regressions. Counts/artist balance were
unchanged (20,953 / 24,598 / target 74 / 127 boosted).

The first deployment was reverted because the strict "halve every term" gate
was not met. It was then adopted by explicit override: the tightened manifest
is strictly better than the conservative one (same 20,953 base / 24,598
balanced rows and artist balance, PASS safety with 0 NSFW/quote/layout losses,
7/7 regressions) even though `anime-style` and `painterly` fall short of half.
`/home/user/dataset/cleanV2/metadata.jsonl` is the tightened output;
`metadata_before_tighten.jsonl` / `report_before_tighten.json` keep the
conservative version for rollback. The generator is at
`scripts/make_clean_v2.py` (local and remote md5
`1bd8d3e82dd022eae935ccd03f4aede1`). No further strip rules were added; the
remaining boilerplate patterns are listed for a possible broader pass later:

- `The art style is clean with soft shading and a bright, airy color palette.`
  (72x) and `The art style features clean line art and soft cel shading with
  bright, even lighting.` (32x): missing adjectives `airy`, `even`, `glossy`,
  `modern`, `polished`, plus plural hues (`pinks`, `purples`, `whites`).
- Leading frames with composition adjectives: `full-body`, `high-quality`,
  `high-resolution`, `multi-panel`, `upper body`, `shot of`, and
  `illustration in a ... style of/featuring`.
- `The art style is soft and painterly with <style phrase>` variants that
  reference concrete props/backgrounds.

## 5. Before / after examples

From the 1k v4 smoke test (`report.json` `examples`):

1. `000001` medium, Sheya (25 chars removed)
   - Before: `An anime illustration of a woman with light blue hair and red eyes, holding a large blue and gold greatsword behind her head. ...`
   - After: `A woman with light blue hair and red eyes, holding a large blue and gold greatsword behind her head. ...`
2. `000002` long, deyui (38 chars)
   - Before: `A digital anime-style illustration of a girl with short black hair and grey eyes, sitting in a crouched position. ...`
   - After: `A girl with short black hair and grey eyes, sitting in a crouched position. ...`
3. `000002` medium, deyui (31 chars)
   - Before: `An anime-style illustration of a girl with short black hair, grey eyes, and black-framed glasses, ...`
   - After: `A girl with short black hair, grey eyes, and black-framed glasses, ...`
4. `000002` no_artist, deyui (38 chars)
   - Prose kept (artist tag removed only); same `A digital anime-style illustration of` -> `A girl ...` strip.
5. `000007` long, yunkaiming (38 chars)
   - Before: `A digital anime-style illustration of a young woman with long, gradient purple-grey hair, ...`
   - After: `A young woman with long, gradient purple-grey hair, ...`
6. `000007` no_artist, yunkaiming (38 chars)
   - Same strip; no artist tag in the variant, prose retained.
7. `000014` medium, oroborus (32 chars)
   - Before: `A digital anime illustration of a pink-skinned character seen from behind against a white background. ...`
   - After: `A pink-skinned character seen from behind against a white background. ...`
8. `000026` long, eiraku_tee (32 chars)
   - Before: `A digital anime illustration of a topless girl with black hair and dog ears, ... reads 'HIHIKI'. ...`
   - After: `A topless girl with black hair and dog ears, ... reads 'HIHIKI'. ...`
   - Single-quoted visible text and the NSFW term `topless` are preserved.

Also verified on the clean30 row 000001:
`A highly detailed anime-style digital illustration, soft warm dusk lighting,
cozy serene summer evening mood, ...` becomes
`An image, cozy serene summer evening mood, ...`, matching the conservative
design in `docs/caption_deconfounding_plan.md` §2.5.

## 6. Validation

`report.json` validation passed on:

- local 1k: 1,000 base / 1,165 balanced; tag-only short 1,000 / 1,165; 0 empty.
- local full: 20,953 base / 24,598 balanced; tag-only short 20,953 / 24,598;
  0 empty; 0 missing images.
- remote v4 full: 20,953 base / 24,598 balanced; 255 artists + 771 unknown;
  target 74; 127 boosted; tag-only short 20,953 / 24,598; 0 empty; 0 missing;
  0 quoted spans lost.

Hard checks include: original rows first in original order, no `repeat_of` on
base rows, balanced count equals the formula, each boosted artist has exactly
74 rows, every known-artist prose variant contains `artist:<id>`, `short`
equals the exact tag-only composition, absolute `image` paths, and quoted spans
compared before/after.

Independent audit on the local full output: no NSFW term was removed, no layout
item `text`/`bbox`/`label` changed, and no non-caption field changed.

## 7. Remote production sequence

SSH (key and flags required because the old `.130` host key changed):

```bash
SSH="ssh -i $HOME/.ssh/id_ed25519_gpu_host -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null user@<gpu-host>"
SCP="scp -i $HOME/.ssh/id_ed25519_gpu_host -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
```

Copy the generator (and optional deploy helper):

```bash
$SCP scripts/make_clean_v2.py \
  user@<gpu-host>:/home/user/MOSS-Lora-Trainer/scripts/make_clean_v2.py
$SCP scripts/deploy_cleanv2_remote.sh \
  user@<gpu-host>:/home/user/MOSS-Lora-Trainer/scripts/deploy_cleanv2_remote.sh
```

Generate the manifest:

```bash
$SSH 'cd /home/user/MOSS-Lora-Trainer && \
  venv/bin/python scripts/make_clean_v2.py \
    --metadata /home/user/dataset/clean30_top256/final_v4/metadata.jsonl \
    --images-root /home/user/dataset/clean30_top256/final_v4 \
    --output /home/user/dataset/cleanV2'
```

Expected stdout: base 20,953; balanced 24,598; extra 3,645; target 74; 127
boosted; tag-only short base 20,953 / balanced 24,598; empty 0; validation PASS.
The generator writes `metadata.jsonl` before returning, so a failed strict
validation still replaces the manifest. Re-running `generate` also overwrites
the deployed manifest: stop the cache and rebuild from scratch after any
manifest change; never swap `metadata.jsonl` while a cache build is running.

Build the **new** cache without touching the old one. The deploy helper
contains the exact `moss_lora.cache` arguments; it checks free disk, refuses to
reuse an existing cache directory, starts the build with `ulimit -n 65536`, and
returns immediately:

```bash
$SSH 'cd /home/user/MOSS-Lora-Trainer && bash scripts/deploy_cleanv2_remote.sh cache'
```

Equivalent direct command (for reference):

```bash
cd /home/user/MOSS-Lora-Trainer
nohup setsid bash -c 'ulimit -n 65536; exec venv/bin/python -m moss_lora.cache \
  --input /home/user/dataset/cleanV2/metadata.jsonl \
  --output /home/user/cache/cleanV2_1m \
  --model /home/user/models/MOSS-Image-Base \
  --target-area 1048576 --max-text-length 1024 \
  --negative-prompt "nsfw, worst quality, bad anatomy, watermark, jpeg artifact" \
  --variant-weights original=0.16,long=0.2133,medium=0.2133,short=0.25,layout=0.20 \
  --missing-caption error' >> /home/user/runs/cleanV2_cache.log 2>&1 < /dev/null &
echo $!
```

Position at the time of this report: **PID 33630** (tightened manifest,
md5 `c85b91d3a9d075dfd066afe47d2accad`), log
`/home/user/runs/cleanV2_cache.log`, text-encoder phase. Earlier builds
(PID 28919 on the conservative manifest, PID 32166 during the revert check)
were stopped and their partial outputs removed. The old cache
`/home/user/cache/clean30_v4_1m` (201G) is untouched. `nvidia-smi` is
`/usr/lib/wsl/lib/nvidia-smi` (not on PATH in non-login shells). Disk: 311G
free at restart; new cache ~236G. Stop if free drops below ~20G.

Check progress:

```bash
$SSH 'ps -p $(cat /home/user/runs/cleanV2_cache.pid) -o pid=,etime=,rss=,cmd=; \
  tail -n 20 /home/user/runs/cleanV2_cache.log; \
  du -sh /home/user/cache/cleanV2_1m; df -h /home/user | tail -1'
```

After the cache process exits 0, verify the index and compute exact steps:

```bash
$SSH 'cd /home/user/MOSS-Lora-Trainer && venv/bin/python - <<'"'"'PY'"'"'
import json
from collections import Counter
idx = json.load(open("/home/user/cache/cleanV2_1m/index.json"))
items = idx["items"]
print("items", len(items))
counts = Counter((int(i["bucket_w"]), int(i["bucket_h"])) for i in items)
steps = sum(n // 8 for n in counts.values())
print("buckets", dict(counts))
print("steps_per_epoch", steps)
print("max_steps_4ep", steps * 4)
PY'
```

Do not assume 12,279/12,280; use the printed `steps * 4`.

Create `configs/train_5090_cleanV2_4ep_r64_bs8.yaml` with:

- `model.path: /home/user/models/MOSS-Image-Base`
- `data.cache_dir: /home/user/cache/cleanV2_1m`
- `data.batch_size: 8`, `drop_last: true`
- `train.output_dir: /home/user/runs/cleanV2_4ep_r64_bs8`
- `train.max_steps: <steps_per_epoch * 4>`
- `lr 1.5e-4`, rank 64, alpha 64, same targets/sampling as the v4 bs8 config
- `save_epoch: true`, `save_every: 3000`, `sample_every: 3000`
- `sample_val_ids` unchanged (`item_00000`, `item_03000`, ... `item_20000`)
- `data.variant_weights` may be empty; `moss_lora.dataset.pick_variant` uses
  the per-item weights stored in the cache index. Verify:

```bash
$SSH 'cd /home/user/MOSS-Lora-Trainer && venv/bin/python - <<'"'"'PY'"'"'
import json
idx = json.load(open("/home/user/cache/cleanV2_1m/index.json"))
item = idx["items"][0]
print(item["variant_kinds"], item["variant_weights"])
assert abs(sum(item["variant_weights"]) - 1.0) < 1e-6
PY'
```

Launch only after the cache exits 0 and the index has 24,598 items:

```bash
$SSH 'cd /home/user/MOSS-Lora-Trainer && mkdir -p /home/user/runs/cleanV2_4ep_r64_bs8 && \
  nohup setsid bash -c '"'"'ulimit -n 65536; cd /home/user/MOSS-Lora-Trainer; \
    export PYTHONUNBUFFERED=1; \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
    exec venv/bin/python -m moss_lora.trainer \
      --config configs/train_5090_cleanV2_4ep_r64_bs8.yaml'"'"' \
    >> /home/user/runs/cleanV2_4ep_r64_bs8.log 2>&1 < /dev/null & echo $!'
```

Then record the PID, check
`/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`
and the first trainer log lines (`dataset_len 24598`, step logs). Do not run the
cache and training concurrently.

The deploy helper automates the same sequence:

```bash
bash scripts/deploy_cleanv2_remote.sh plan
bash scripts/deploy_cleanv2_remote.sh generate
bash scripts/deploy_cleanv2_remote.sh cache
bash scripts/deploy_cleanv2_remote.sh finalize   # after cache exits 0
bash scripts/deploy_cleanv2_remote.sh train
```
