# iapi：Qwen Image 图片生成与专业照片修复服务

基于 FastAPI、Redis、MySQL/SQLite 和 GPU Worker 的 OpenAI 风格图片 API。支持 Qwen Image 2512、Qwen Image Edit、Real-ESRGAN、SwinIR、CodeFormer、InsightFace ArcFace、关键点过滤、人脸软遮罩贴回，以及可选的独立 SUPIR Worker。

> **SUPIR 许可提醒**：SUPIR 官方仓库声明软件仅限非商业使用。商业部署前必须取得作者书面许可；同时确认 SDXL、LLaVA、CodeFormer、InsightFace 等模型和依赖的许可条件。

## 1. 架构

```text
Client → API (:8000) → Redis queue → GPU Worker(s) → MySQL/SQLite
                                      ├─ Qwen / Real-ESRGAN / SwinIR
                                      ├─ CodeFormer → ArcFace → 关键点过滤
                                      └─ 综合评分 → 软遮罩贴回 → 输出

API/Worker ──HTTP──► 独立 SUPIR Worker (:8010, optional)
```

生产部署固定为三类节点：API 机器（API + Redis/MySQL 客户端）、4×4090 机器（一个 Qwen 四卡分层 Worker）、另一台 GPU 机器（独立 SUPIR Worker）。单机开发才使用 `SERVICE_ROLE=all`、memory 队列和 SQLite。

## 2. 环境要求

### API 节点

- Python 3.10+，建议 3.11
- 无需 GPU
- 生产使用 Redis、MySQL
- 安装 `requirements-api.txt`

### Qwen GPU Worker

- Linux + NVIDIA Driver/CUDA
- 4×4090 分层加载时只启动一个 Worker 进程；不要启动四个单卡 Worker
- 安装 `requirements-worker.txt`

### SUPIR Worker

- 独立 NVIDIA GPU 主机
- 推荐至少 16GB 显存；官方低显存方式也约需 12GB Diffusion 显存
- Docker、Docker Compose v2、NVIDIA Container Toolkit
- 详细部署见 `deploy/supir/README.md`

本开发机为 MX450 2GB，无法真实运行 SUPIR；不能使用本机结果作为 SUPIR 基准。

## 3. 所有环境配置模板

| 文件 | 用途 | 启动方式 |
|---|---|---|
| `.env.example` | 单机 API + Worker 开发 | `uvicorn app.main:app --reload` |
| `.env.api.example` | 生产 API-only 节点 | `uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| `.env.worker.example` | 生产 GPU Worker 节点 | `python -m app.worker` |
| `deploy/supir/.env.example` | 独立 SUPIR Worker | `docker compose up -d` |

Windows PowerShell：

```powershell
Copy-Item .env.example .env
Copy-Item deploy/supir/.env.example deploy/supir/.env
```

Linux：

```bash
cp .env.example .env
cp deploy/supir/.env.example deploy/supir/.env
```

不要提交真实 `.env`。环境变量名称由 `app/config.py` 和 `supir_worker/settings.py` 定义，使用大写下划线形式。

## 4. 单机开发部署

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Windows 使用 Copy-Item；Linux 使用 cp
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`.env` 建议：

```env
SERVICE_ROLE=all
TASK_QUEUE_BACKEND=memory
TASK_DB_BACKEND=sqlite
TASK_DB_PATH=data/image_tasks.sqlite3
OUTPUT_DIR=outputs
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

## 5. 生产三节点部署

### 5.1 共享服务

先准备 Redis 和 MySQL 数据库。API 与所有 Worker 必须完全一致：

- `REDIS_URL`
- `REDIS_QUEUE_NAME`
- `REDIS_PROCESSING_QUEUE_NAME`
- `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_DATABASE` 及账号

服务会自动创建任务表，但不会自动创建 MySQL 数据库。

### 5.2 API 节点

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
cp .env.api.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

关键配置：

```env
SERVICE_ROLE=api
TASK_QUEUE_BACKEND=redis
TASK_DB_BACKEND=mysql
REDIS_URL=redis://:password@redis-host:6379/0
MYSQL_HOST=mysql-host
MYSQL_DATABASE=iapi
TASK_PUBLIC_BASE_URL=https://api.example.com
```

### 5.3 4×4090 Qwen Worker 节点

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-worker.txt
cp .env.worker.example .env
python -m app.worker
```

关键配置：

```env
SERVICE_ROLE=worker
TASK_QUEUE_BACKEND=redis
TASK_DB_BACKEND=mysql
WORKER_NAME=qwen-4gpu-node
MODEL_GPU_IDS=0,1,2,3
MODEL_GPU_COUNT=4
MODEL_GPU_MEMORY_LIMIT=
DEVICE=auto
QWEN_EDIT_MULTI_GPU_ENABLED=true
QWEN_EDIT_TRANSFORMER_SHARDING_ENABLED=true
QWEN_EDIT_DEVICE_MAP=balanced
IMAGE_WORKER_COUNT=1
REDIS_URL=redis://:password@redis-host:6379/0
MYSQL_HOST=mysql-host
```

该节点只启动一个进程：

```bash
python -m app.worker
```

不要使用 `CUDA_VISIBLE_DEVICES=0`，也不要启动四个 Worker。四张卡共同承载同一个 Qwen 模型，`IMAGE_WORKER_COUNT=1` 保证一次只处理一个 GPU 任务。

如果使用 `CUDA_VISIBLE_DEVICES`，必须让进程看到全部四张卡，例如 `CUDA_VISIBLE_DEVICES=0,1,2,3`；通常直接使用 `MODEL_GPU_IDS=0,1,2,3` 即可。

### 5.4 三节点连接关系

```text
API 机器
  ├─ HTTP :8000
  ├─ Redis 客户端 ─────────────┐
  └─ MySQL 客户端 ─────────────┤
                              ▼
4×4090 机器：一个 Qwen 四卡分层 Worker
                              │
                              └─ HTTP ──► 独立 SUPIR Worker（另一台 GPU 机器）
```

Qwen Worker 和 SUPIR Worker 不共享 GPU。Qwen Worker 的 `.env` 配置：

```env
SUPIR_ENABLED=true
SUPIR_BASE_URL=http://supir-host:8010
SUPIR_API_KEY=与SUPIR Worker相同的随机密钥
SUPIR_TIMEOUT=900
SUPIR_ENDPOINT=/v1/restore
```

不要在 4×4090 机器上再启动 SUPIR 进程，避免 SUPIR 与 Qwen 分层模型争用显存。

## 6. 模型准备

| 功能 | 配置 | 权重来源/行为 |
|---|---|---|
| Qwen Image 2512 | `QWEN_IMAGE_MODEL_PATH` | Hugging Face `Qwen/Qwen-Image-2512` |
| Qwen Edit 2511 | `QWEN_EDIT_MODEL_PATH` | Hugging Face `Qwen/Qwen-Image-Edit-2511` |
| Unblur LoRA | `QWEN_UNBLUR_UPSCALE_LORA_PATH` | Hugging Face，按需下载 |
| Real-ESRGAN | `REALESRGAN_MODEL_PATH` | 官方 `.pth`，为空时下载到 `weights/` |
| SwinIR | `SWINIR_MODEL_PATH` | 按需下载到 `weights/SwinIR` |
| CodeFormer | `CODEFORMER_MODEL_PATH` | 官方 `codeformer.pth`，按需下载 |
| InsightFace | `INSIGHTFACE_MODEL_ROOT` | `buffalo_l` 模型包按需下载 |
| SUPIR | `deploy/supir/.env` | 需要 SDXL、SUPIR-v0Q/v0F 官方权重 |

生产建议预下载权重、固定版本和本地路径，不要让多个 Worker 同时首次下载同一个大文件。

## 7. 专业照片修复

请求使用 `enhance_mode=restoration`：

| 模式 | 行为 |
|---|---|
| `preserve` | RealESRNet，保守放大，不做人脸重建 |
| `balanced` | SwinIR（启用时）→ Real-ESRGAN → 人脸候选评估 |
| `creative` | 可使用 Qwen Edit 和独立 SUPIR |
| `auto` | 根据模糊、噪声、块效应和细节选择 preserve/balanced |

`auto` 还会根据颜色量化误差、强边缘比例和饱和度估算动漫/插画风格。命中后使用 `RealESRGAN_x4plus_anime_6B`，并跳过面向照片的 SwinIR 与 CodeFormer/ArcFace 人脸链路，避免动漫五官被照片模型重建。

人脸候选顺序：CodeFormer → ArcFace → 五点关键点形变过滤 → 综合评分 → 软遮罩贴回。任何候选不安全都会自动回退原始人脸。

重要配置：

```env
SWINIR_ENABLED=true
CODEFORMER_ENABLED=true
INSIGHTFACE_ENABLED=true
INSIGHTFACE_IDENTITY_THRESHOLD=0.65
LANDMARK_DEFORMATION_RMS_THRESHOLD=0.08
LANDMARK_DEFORMATION_MAX_THRESHOLD=0.16
FACE_MASK_INSET_RATIO=0.12
FACE_MASK_BLUR_RATIO=0.08
FACE_MASK_OPACITY=0.90
FACE_CANDIDATE_MIN_SCORE=0.72
FACE_CANDIDATE_IDENTITY_WEIGHT=0.50
FACE_CANDIDATE_GEOMETRY_WEIGHT=0.30
FACE_CANDIDATE_QUALITY_WEIGHT=0.15
FACE_CANDIDATE_DETECTION_WEIGHT=0.05
RESTORATION_ANIME_DETECTION_ENABLED=true
RESTORATION_ANIME_SCORE_THRESHOLD=0.72
RESTORATION_ANIME_REALESRGAN_MODEL_NAME=RealESRGAN_x4plus_anime_6B
```

## 8. 独立 SUPIR Worker

SUPIR 源码、权重和 GPU 必须位于独立主机：

```bash
cd deploy/supir
cp .env.example .env
# 修改 SUPIR_MODELS_DIR 和 SUPIR_WORKER_API_KEY
docker compose up --build -d
curl http://127.0.0.1:8010/health
```

主 iapi Worker：

```env
SUPIR_ENABLED=true
SUPIR_BASE_URL=http://supir-host:8010
SUPIR_API_KEY=与独立 Worker 相同的随机密钥
SUPIR_TIMEOUT=900
SUPIR_ENDPOINT=/v1/restore
```

SUPIR Worker 使用 Bearer Token，不应直接暴露到公网。官方 Q/F 权重必须从官方 README 提供的地址获取，不使用来源不明的重打包权重。

## 9. 真实图片基准测试

准备真实图片目录，例如 `/root/iapi/images`，不要使用纯色或合成图作为质量结论：

```bash
python scripts/benchmark_supir.py \
  --input-dir /root/iapi/images \
  --output-dir benchmark-results/supir-q \
  --base-url http://supir-host:8010 \
  --api-key "$SUPIR_WORKER_API_KEY" \
  --upscale 2
```

输出：恢复图片、`records.csv`、`summary.json`。指标包括成功率、平均/P50/P95 延迟、百万像素吞吐、细节增益和像素 MAE。MAE 仅是保真度代理指标，正式验收还应进行人眼盲测、身份一致性和文字保真检查。

## 10. API 示例

提交专业修复并查询任务：

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits \
  -F "prompt=自然修复照片，保持人物身份、文字和构图" \
  -F "image=@input.png" \
  -F "enhance_mode=restoration" \
  -F "restoration_mode=auto" \
  -F "resolution=4k"

curl http://127.0.0.1:8000/v1/images/tasks/<task_id>
```

使用 URL 图片进行图生图：

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-image-2512",
    "prompt": "一只戴帽子的猫",
    "image": "https://example.com/input.png"
  }'
```

`image` 支持 `http://`/`https://` 图片 URL、Base64、Data URL 和 multipart 文件。URL 必须能被 API/Worker 服务端直接访问，并返回有效图片内容。

文字、Logo 和排版保真优先使用 `enhance_mode=pixel` 或 `realesrgan`；扩散模型不提供像素级文字不变保证。

## 11. 测试与排障

```bash
python -m pytest -q
```

- API 入队成功但 Worker 不执行：检查 Redis 三项队列配置是否一致。
- Redis 空等超时：`REDIS_SOCKET_TIMEOUT` 必须大于 `REDIS_BLOCK_TIMEOUT`。
- GPU OOM：降低图片尺寸、采样步数、tile，并确保每卡只有一个 Worker。
- SwinIR 返回 `HTTP Error 404`：旧版本使用了错误的权重名前缀；当前代码会使用官方 `005_colorDN_DFWB...` 和 `006_colorCAR_DFWB...` 文件名。更新代码后删除不完整的 `weights/SwinIR/*.part` 文件再重试。
- CodeFormer 报 `No module named 'spandrel_extra_arches'`：在 GPU Worker 环境执行 `pip install -r requirements-worker.txt`；不要只在 API 机器安装该依赖。
- SwinIR 报 `attn_mask size mismatch`：说明模型结构参数与官方权重不一致。当前彩色降噪模型必须使用 `img_size=128`、`window_size=8`；JPEG 彩色模型必须使用 `img_size=126`、`window_size=7`。
- 人脸未贴回：查看 `restoration_face_scores`、`rejection_reason`、`restoration_faces_pasted`。
- SUPIR 503：检查 GPU、模型路径、容器日志和 `/health`；2GB 显存无法运行。

## 12. 安全清单

- 不提交 `.env`、API Key、MySQL 密码、OSS 密钥或模型凭证。
- API、Redis、MySQL、SUPIR Worker 使用内网和防火墙限制。
- 生产使用 MySQL + Redis + OSS，不依赖本地 SQLite 和本地输出目录。
- 固定模型与源码版本，保留基准报告、GPU、CUDA 和依赖信息。
- 启用 SUPIR 前确认非商业许可；商业用途先取得书面授权。
