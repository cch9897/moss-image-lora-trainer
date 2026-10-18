# 实时训练看板

`moss-lora-dashboard` 是一个**只读**的本地 Web 看板，用来监控一次正在进行的 LoRA 训练：loss / 学习率曲线、验证采样图、checkpoint、配置、ETA 和自动刷新。它直接读取训练器已经写出的 run 目录，不介入训练进程，也不会提供权重下载。

看板没有前端构建步骤、没有 CDN 或外部字体：`moss_lora/dashboard.html` 内含全部 CSS/JS 与手写 SVG 图表。

## 1. 安装与启动

FastAPI/uvicorn 是可选依赖：

```bash
source .venv/bin/activate
pip install -e '.[dashboard]'
```

启动（默认监听 `127.0.0.1:7860`，扫描 `runs/`）：

```bash
moss-lora-dashboard
# 等价于
moss-lora-dashboard --root runs --host 127.0.0.1 --port 7860 --refresh 5 --stale-after 300
```

浏览器打开 <http://127.0.0.1:7860/>。`--run moss_1m` 可以预选一个 run；`--open` 会在服务启动后自动打开浏览器。

也可以不安装 console script 直接运行：

```bash
python -m moss_lora.dashboard --root runs --port 7860
```

### 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--root` | `runs` | 扫描 run 的根目录；根目录本身也可以是 run |
| `--host` | `127.0.0.1` | 绑定地址；默认只允许本机访问 |
| `--port` | `7860` | 端口 |
| `--refresh` | `5` | 浏览器轮询间隔（秒） |
| `--stale-after` | `300` | metrics 超过多少秒未更新后标记为 `stale` |
| `--open` | 关闭 | 启动后打开系统浏览器 |
| `--run` | 无 | 初始选中的 run id |

## 2. 看板内容

- **Header**：run 选择器、状态徽标、当前 step / max_steps、ETA、最后一次日志时间、刷新间隔。
- **Loss tape**：header 下沿的签名元素——最近 80 个记录步的竖线，高度与颜色随 loss 变化，最新一步在右侧。
- **Stats strip**：latest loss、20 步 EMA、学习率、sigma、grad norm、s/step、峰值显存。
- **Loss 图**：原始 loss 曲线（vermilion 实线）+ EMA（墨色虚线），悬停显示十字线与 tooltip。
- **Learning rate 图**：蓝色 LR 曲线，同样支持悬停。
- **Samples**：按 `step_*` 分组，最新在前；点击图片打开灯箱。无 VAE 时训练器写出的 `*.latent.pt` 显示为 `latent` 占位块，不提供预览。
- **Checkpoints**：step、adapter 大小、optimizer 大小、写入时间；**不提供权重下载**。
- **Configuration**：rank/alpha、lr、batch、max_steps、cache_dir、base_model、schedule、targets。

自动刷新会保留当前 run 和选中的 sample step。请求失败时页面保留上一次成功的数据，并在页头标记 `offline · retrying`。

空目录会显示可直接复制的启动命令：

```bash
moss-lora-train --config configs/train_5090_1m.yaml
```

## 3. 数据来源

看板只读取以下文件（全部由训练器产生）：

| 路径 | 用途 |
| --- | --- |
| `metrics.csv` | loss / lr / sigma / grad_norm / sec_per_step / peak_vram_gb；允许边写边读 |
| `trainer_state.json` | run 根目录优先；不存在时读取最新 checkpoint 中的同名文件 |
| `adapter_config.json` | run 根目录优先；否则读取 checkpoint 中的文件，用于 rank/alpha/targets |
| `samples/step_*/` | `.png` / `.jpg` / `.jpeg` / `.webp` 提供预览；`.pt` 仅列出 |
| `checkpoints/step_*/` | 只列出文件名、大小和时间 |

`metrics.csv` 的解析是容错的：追加过程中产生的不完整行、列数不符、非数字字段会被跳过；同一 step 重复出现时保留最后一行，并按 step 排序。loss 的 EMA 使用 `alpha = 2 / (20 + 1)`。曲线在服务端降采样到最多 1200 点，并始终保留最后一个点。

### 状态与 ETA

| 状态 | 条件 |
| --- | --- |
| `finished` | `step >= max_steps` |
| `running` | metrics 最近更新时间不超过 `--stale-after` |
| `stale` | 超过 `--stale-after` 没有新 metrics |
| `unknown` | 还没有可解析的 metrics 行 |

ETA 仅在 `running` 且 `max_steps` 已知时计算：`(max_steps - step) × median(最近 20 条 sec_per_step)`。

### Run 发现

- 根目录下任意包含 `metrics.csv` 的目录都算一个 run（递归深度最多 4 层）。
- run id 是相对根目录的 POSIX 路径，例如 `moss_1m`、`project/second`；如果根目录本身就是 run，id 为 `.`。
- 列表按 `metrics.csv` 的修改时间倒序。
- 扫描会跳过 `.git`、`.venv`、`__pycache__`、`node_modules`、`samples`、`checkpoints`。

## 4. HTTP API

所有响应带 `Cache-Control: no-store`。

| 端点 | 说明 |
| --- | --- |
| `GET /` | 看板 HTML |
| `GET /api/runs` | `{root, runs: [...]}` 概览 |
| `GET /api/run?path=<run_id>` | 完整详情（配置、stats、metrics、samples、checkpoints、ETA） |
| `GET /media?run=<run_id>&file=samples/step_.../x.png` | 受控图片响应 |
| `GET /healthz` | `{"status": "ok"}` |

`/api/runs` 中每个 run 包含：

```json
{
  "id": "moss_1m",
  "name": "moss_1m",
  "step": 640,
  "max_steps": 1200,
  "latest_loss": 0.0989,
  "latest_lr": 0.00006922,
  "updated_at": "2026-10-18T09:00:00Z",
  "status": "running",
  "sample_count": 21,
  "checkpoint_count": 3
}
```

错误统一返回 JSON：

```json
{"error": "Run not found: missing"}
```

## 5. 安全边界

- 只读：不写入 run 目录，不启动、停止或修改训练。
- 不提供权重：`/media` 只接受 `.png`、`.jpg`、`.jpeg`、`.webp`，且解析后的真实路径必须位于该 run 目录内。
- 拒绝绝对路径、`..`、反斜杠和逃逸根目录的符号链接。
- run id 同样拒绝绝对路径、`..` 和逃逸符号链接。
- 默认绑定 `127.0.0.1`，没有身份认证。需要远程访问时使用 SSH 端口转发，不要把端口直接暴露到公网。

## 6. 通过 SSH 端口转发查看

在训练机器上启动（保持默认的本机绑定）：

```bash
moss-lora-dashboard --root runs --port 7860
```

在本地电脑建立隧道：

```bash
ssh -N -L 7860:127.0.0.1:7860 user@training-host
```

然后打开 <http://127.0.0.1:7860/>。`-N` 只转发端口、不打开远端 shell；也可以直接用 `ssh -L 7860:127.0.0.1:7860 user@host` 登录。

如果本机 7860 已被占用，把两边一起改掉，例如：

```bash
moss-lora-dashboard --port 7861
ssh -L 7861:127.0.0.1:7861 user@training-host
```

避免使用 `--host 0.0.0.0`，除非你已经在可信网络内加了反向代理与访问控制。

## 7. 排错

| 现象 | 处理 |
| --- | --- |
| `Install them with: pip install -e '.[dashboard]'` | 按提示安装可选依赖 |
| 看板显示 `No runs found` | 确认 `--root` 指向包含 `metrics.csv` 的目录；训练启动后 metrics 才会出现 |
| run 一直是 `stale` | 训练可能已停止，或 `--stale-after` 太小；用日志间隔估算合理值 |
| 没有 samples | 需要 `logging.sample_every > 0` 且 `logging.sample_val_ids` 非空；无 VAE 时只有 `.latent.pt` |
| 没有 checkpoints | 训练尚未达到 `logging.save_every` |
| `Configuration appears after the first checkpoint` | 目前 `trainer_state.json` 写在 checkpoint 内；也可以手动在 run 根目录放一份 |
| 曲线点数少于 step 数 | `metrics.csv` 按 `log_every` 记录；服务端还会降采样到 1200 点 |
| 端口无法访问 | 确认没有把服务绑定到非本机地址，以及 SSH 转发端口一致 |
