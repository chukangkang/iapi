# FLUX.2 Klein KV OpenAI Image API

一个基于 FastAPI + 🧨 Diffusers 的 OpenAI 风格图片接口服务，默认使用 `black-forest-labs/FLUX.2-klein-9b-kv` 和 `Flux2KleinKVPipeline`。

## 接口

### `POST /v1/images/generations`

异步提交图片生成任务，立即返回任务 ID。后台 worker 会按队列执行 GPU 推理。

支持：

- 文生图（t2i）：只传 `prompt`
- 图生图（i2i）：传 `prompt` + `image`

`image` 支持：

- JSON 字段中的图片 URL
- JSON 字段中的 base64 / data URL
- multipart 表单文件

常用参数：

| 字段 | 说明 | 默认值 |
| --- | --- | --- |
| `prompt` | 提示词，必填 | - |
| `negative_prompt` | 反向词；会与 `.env` 的 `DEFAULT_NEGATIVE_PROMPT` 去重合并，底层 pipeline 支持时生效 | `.env` 默认值 |
| `image` | 可选参考图；存在时走 i2i/KV cache 流程 | `null` |
| `size` | OpenAI 风格尺寸，如 `768x768`、`1024x1024` | `768x768` |
| `aspect_ratio` | 主流比例预设：`16:9`、`4:3`、`1:1`、`9:16` | `null` |
| `resolution` | 搭配 `aspect_ratio` 使用：`2k` 或 `4k` | `2k` |
| `width` / `height` | 显式宽高，会覆盖 `size` 对应维度 | `.env` 默认值 |
| `num_inference_steps` | 推理步数 | `4` |
| `seed` | 随机种子 | `null` |
| `response_format` | `url` 或 `b64_json` | `url` |
| `enhance_mode` | 高清/保真模式：`flux`、`pixel`、`realesrgan`、`realesrgan_flux`、`qwen_edit`、`qwen_edit_realesrgan`、`qwen_unblur_upscale`、`qwen_unblur_upscale_realesrgan` | `.env` 默认值 |
| `flux_refine_strength` | 图生图时传给 FLUX 的低重绘强度；pipeline 不支持时会自动忽略 | `0.08` |
| `n` | 当前仅支持 `1` | `1` |

### `POST /v1/images/edits`

异步提交图片编辑任务，立即返回任务 ID。后台 worker 会按队列执行 GPU 推理。

支持 `application/json` 图片 URL/base64，也支持 multipart 表单文件上传，字段与 OpenAI 图片编辑接口接近：

- `prompt`：必填
- `image`：必填，JSON 中可传图片 URL/base64，multipart 中可传文件
- `mask`：可选；当前接口会接收但 FLUX.2 Klein KV 示例未使用 mask，因此暂不参与推理
- 其他参数同 generations

### `POST /v1/chat/completions`

用于兼容 New API / OpenAI 后端连通性测试。该接口不会调用大语言模型，只返回当前后端状态和 `.env` 中配置的 `MODEL_NAME`，证明服务可正常访问。

### `GET /v1/models`

用于兼容 New API / OpenAI 模型列表探测，返回 `.env` 中配置的 `MODEL_NAME`。

## 主流比例

你可以继续用 OpenAI 风格的 `size="宽x高"`，也可以使用 `aspect_ratio` + `resolution` 请求主流尺寸：

| 场景 | `aspect_ratio` | `resolution=2k` | `resolution=4k` |
| --- | --- | --- | --- |
| 视频、海报、桌面图 | `16:9` | `2560x1440` | `3840x2160` |
| 老照片、证件、老式图片 | `4:3` | `2048x1536` | `4096x3072` |
| 方形图、头像、朋友圈 | `1:1` | `1440x1440` | `4096x4096` |
| 手机竖图、短视频封面 | `9:16` | `1440x2560` | `2160x3840` |

为了避免 4090 24G 在 2K/4K 直接推理时 OOM，`enhance_mode=flux` 会根据 `.env` 里的 `MAX_GENERATION_PIXELS` 先按比例生成安全尺寸，再放大到目标输出尺寸。默认 `786432` 约等于 `1024x768` 的像素预算。

如果目标是视频高清、海报高清、字幕/商品图文字保真，建议不要用默认 FLUX 重绘，而是使用 `enhance_mode=pixel` 或 `enhance_mode=realesrgan`。

## 安装

建议使用 Python 3.11+。生产拆分部署时，API 节点只需要轻量 Web/数据库依赖；Worker 节点才需要 CUDA / PyTorch / Diffusers 等 GPU 推理依赖。

API 节点安装：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-api.txt
```

Worker 节点安装：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-worker.txt
```

单机开发或 `SERVICE_ROLE=all` 可直接安装全量依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 保持为全量兼容安装；`requirements-api.txt` 用于 API-only 节点；`requirements-worker.txt` 会包含 API 基础依赖并追加 GPU/模型相关依赖。

如果模型需要 Hugging Face 访问权限，请编辑 `.env`：

```env
MODEL_NAME=flux-image-backend
HF_TOKEN=你的 HuggingFace Token
TOKENIZERS_PARALLELISM=false
```

`MODEL_NAME` 是专门返回给 New API / OpenAI 兼容测试的模型名，默认 `flux-image-backend`；实际加载的 Hugging Face 模型仍由 `MODEL_PATH` 控制。

`TOKENIZERS_PARALLELISM=false` 用于关闭 Hugging Face `tokenizers` 在多进程/fork 场景下的并行 warning，不影响图片生成结果。

### 模型与下载链接

部署前需要准备的模型/权重如下：

| 用途 | 配置项 | 默认值 / 文件名 | 下载地址 | 是否必需 |
| --- | --- | --- | --- | --- |
| FLUX 文生图/图生图基座 | `MODEL_PATH` | `black-forest-labs/FLUX.2-klein-9b-kv` | https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv | 使用 `flux` / 默认生成时必需 |
| Qwen 图片编辑基座 | `QWEN_EDIT_MODEL_PATH` | `Qwen/Qwen-Image-Edit-2511` | https://huggingface.co/Qwen/Qwen-Image-Edit-2511 | 使用 `qwen_edit*` / `qwen_unblur_upscale*` 时必需 |
| Qwen 去模糊高清 LoRA | `QWEN_UNBLUR_UPSCALE_LORA_PATH` | `prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale` | https://huggingface.co/prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale | 使用 `qwen_unblur_upscale*` 时必需 |
| Qwen LoRA 权重文件 | `QWEN_UNBLUR_UPSCALE_LORA_WEIGHT_NAME` | `Qwen-Image-Edit-Unblur-Upscale_20.safetensors` | https://huggingface.co/prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale/tree/main | 使用 `qwen_unblur_upscale*` 时必需 |
| Real-ESRGAN 通用超分 | `REALESRGAN_MODEL_NAME` | `realesr-general-x4v3.pth` | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth | 使用 `realesrgan*` 时必需 |
| Real-ESRGAN 可选降噪权重 | 同目录自动识别 | `realesr-general-wdn-x4v3.pth` | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth | 可选，配合 `REALESRGAN_DENOISE_STRENGTH` 混合 |

Hugging Face 模型可直接填仓库 ID，Diffusers 会在首次加载时自动下载到本机缓存；如果服务器不能联网，可提前下载到本地目录，并把对应配置项改成本地路径。Real-ESRGAN 当前代码只加载 `.pth` 权重，需手动下载到 `REALESRGAN_MODEL_PATH` 指向的目录，或直接把 `REALESRGAN_MODEL_PATH` 指向具体 `.pth` 文件。

### 高清保真模式

`.env` 可配置：

```env
DEFAULT_ENHANCE_MODE=flux
DEFAULT_NEGATIVE_PROMPT=extra hands, extra fingers, malformed hands, bad hands, fused fingers, missing fingers, extra arms, deformed limbs, mutated body, bad anatomy, disfigured, distorted face, low quality
FLUX_REFINE_STRENGTH=0.08
QWEN_EDIT_MODEL_PATH=Qwen/Qwen-Image-Edit-2511
QWEN_EDIT_PIPELINE_CLASS=QwenImageEditPlusPipeline
QWEN_EDIT_STEPS=10
QWEN_EDIT_GUIDANCE_SCALE=1.0
QWEN_EDIT_TRUE_CFG_SCALE=4.0
QWEN_EDIT_STRENGTH=0.7
QWEN_EDIT_SCALE_TO_SIDE=longest
QWEN_EDIT_SCALE_TO_LENGTH=2048
QWEN_EDIT_ROUND_TO_MULTIPLE=16
QWEN_EDIT_BACKGROUND_COLOR=#000000
QWEN_EDIT_QUANTIZATION=none
QWEN_EDIT_DEVICE_MAP=balanced
QWEN_UNBLUR_UPSCALE_LORA_PATH=prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale
QWEN_UNBLUR_UPSCALE_LORA_WEIGHT_NAME=Qwen-Image-Edit-Unblur-Upscale_20.safetensors
QWEN_UNBLUR_UPSCALE_TRIGGER_PROMPT=unblur and upscale
QWEN_UNBLUR_UPSCALE_LORA_SCALE=1.0
PIXEL_SHARPEN_ENABLED=true
PIXEL_SHARPEN_RADIUS=1.4
PIXEL_SHARPEN_PERCENT=140
PIXEL_SHARPEN_THRESHOLD=3
UPSCALE_FIT_MODE=cover
UPSCALE_FILL_COLOR=black
REALESRGAN_MODEL_PATH=
REALESRGAN_MODEL_NAME=realesr-general-x4v3.pth
REALESRGAN_MAX_PASSES=2
REALESRGAN_DENOISE_STRENGTH=0.35
REALESRGAN_TILE=512
```

`DEFAULT_NEGATIVE_PROMPT` 是全局默认反向词。当前端不传 `negative_prompt` 时会直接使用该配置；当前端传入时，服务会按英文逗号、中文逗号、分号和换行拆分后去重合并，避免重复堆叠相同反向词。

模式说明：

| 模式 | 说明 | 文字一致性 |
| --- | --- | --- |
| `flux` | FLUX 文生图/图生图；图生图会按 `flux_refine_strength` 低强度参考原图重绘 | 可能改变文字 |
| `pixel` | Lanczos 像素放大 + 可配置锐化，不进扩散模型 | 最稳定 |
| `realesrgan` | 先用 Real-ESRGAN 多轮超分覆盖目标尺寸，再缩放到 4K，不进 FLUX | 高 |
| `realesrgan_flux` | 先 Real-ESRGAN，再尝试 FLUX 极低强度细节修复 | 仍可能轻微改变 |
| `qwen_edit` | Python 直接加载 Qwen Image Edit 做高清编辑，再像素放大到目标尺寸 | 中 |
| `qwen_edit_realesrgan` | Qwen Image Edit 高清编辑后，再用 Real-ESRGAN 输出目标尺寸 | 中高 |
| `qwen_unblur_upscale` | 加载 `Qwen-Image-Edit-2511-Unblur-Upscale` LoRA 做去模糊/高清增强，再像素放大到目标尺寸 | 中高 |
| `qwen_unblur_upscale_realesrgan` | Qwen Unblur/Upscale LoRA 修复后，再用 Real-ESRGAN 输出目标尺寸 | 中高 |

如果追求“看起来更高清”，可用 `enhance_mode=flux` 直接让 FLUX 参考原图低强度重绘到 4K；如果严格要求“字体、文字 100% 不变”，优先 `pixel` 或 `realesrgan`。`pixel` 只是保真放大和锐化，不会凭空生成新纹理；`realesrgan` 是 AI 超分细节增强，且不会进入 FLUX 重绘；`qwen_unblur_upscale`、`flux` 和 `realesrgan_flux` 都会进入修复/扩散模型，没有像素级一致性保证。

如果想接近 ComfyUI 中 “Qwen Image Edit 低步数修复 + 高清放大” 的效果，不需要调用 ComfyUI 服务，可使用 `enhance_mode=qwen_edit`、`enhance_mode=qwen_edit_realesrgan`、`enhance_mode=qwen_unblur_upscale` 或 `enhance_mode=qwen_unblur_upscale_realesrgan`。服务会在 Python 内部加载 `QWEN_EDIT_PIPELINE_CLASS` 指定的 Diffusers pipeline，先按 `QWEN_EDIT_SCALE_TO_SIDE` / `QWEN_EDIT_SCALE_TO_LENGTH` 计算编辑尺寸，再输出到目标 4K。

`QWEN_EDIT_TRUE_CFG_SCALE=4.0` 会在 pipeline 支持 `true_cfg_scale` 时自动传入，更贴近 Qwen Image Edit 2511 / Plus pipeline 的推荐调用方式；老 pipeline 不支持时会自动忽略。`QWEN_EDIT_STEPS` 建议从 `10` 起步，画质通常比 `4` 步稳定。

`qwen_unblur_upscale` 使用 Hugging Face 上的 `prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale` LoRA。它不是完整模型，而是加载到 `Qwen/Qwen-Image-Edit-2511` 基座上的适配器；默认权重文件是作者推荐的 `Qwen-Image-Edit-Unblur-Upscale_20.safetensors`。`QWEN_UNBLUR_UPSCALE_TRIGGER_PROMPT` 会自动拼到用户请求的 `prompt` 前面以触发 LoRA，不会覆盖用户提示词；如果请求提示词里已经包含触发词，则不会重复拼接。如果想更弱一点，可把 `QWEN_UNBLUR_UPSCALE_LORA_WEIGHT_NAME` 改成 `_15.safetensors` 或 `_10.safetensors`。

截图中的 Qwen 工作流尺寸参数对应这里的 `QWEN_EDIT_SCALE_TO_SIDE=longest`、`QWEN_EDIT_SCALE_TO_LENGTH=2048`、`QWEN_EDIT_ROUND_TO_MULTIPLE=16`、`QWEN_EDIT_BACKGROUND_COLOR=#000000`。服务会先把参考图按最长边缩放到 2048，再 letterbox 到 16 的倍数尺寸，送入 Qwen Edit；最终再输出到请求的 4K 尺寸。

4090 24G 显存紧张时可开启 Qwen Edit 量化：`QWEN_EDIT_QUANTIZATION=8bit`。更省显存可试 `4bit`，但速度和画质可能波动；量化依赖 Linux 下的 `bitsandbytes` 和支持 `PipelineQuantizationConfig` 的新版 Diffusers，默认 `none` 不启用。量化加载时 `QWEN_EDIT_DEVICE_MAP` 可选 `balanced`、`cuda`、`cpu`，默认 `balanced`；`8bit + balanced` 会启用 CPU fp32 offload，牺牲速度换稳定加载。若使用已经量化好的 Diffusers 仓库，例如 `ovedrive/Qwen-Image-Edit-2511-4bit`，请保持 `QWEN_EDIT_QUANTIZATION=none`，并设置 `QWEN_EDIT_PIPELINE_CLASS=QwenImageEditPlusPipeline`。

`REALESRGAN_MAX_PASSES=2` 表示最多连续做 2 轮 x4 超分：例如 `396x234` 会先超分到足够覆盖 `3840x2160`，再缩放到目标 4K，避免一次插值硬拉导致模糊。显存紧张时可调为 `1`。

`UPSCALE_FIT_MODE=cover` 会保持原图比例并居中裁剪到目标 4K，避免黑边，也避免强行拉伸导致字体变形。可选值：`cover` 保比例居中裁剪、`contain` 保比例补边、`stretch` 强制拉伸。高清输出推荐保持 `cover`。

当前 `realesrgan` Python 推理包需要 `.pth` 权重，不能直接加载 Hugging Face/Qualcomm 目录里的 `model.safetensors`。推荐下载最新版通用小模型 `realesr-general-x4v3.pth`，文字/视频高清场景先用它；`realesr-general-wdn-x4v3.pth` 是强降噪搭配权重，可选下载到同一目录。

服务器示例：

```env
REALESRGAN_MODEL_PATH=/root/xinglin-data/chat/weights
REALESRGAN_MODEL_NAME=realesr-general-x4v3.pth
REALESRGAN_DENOISE_STRENGTH=0.35
```

下载地址：

- `realesr-general-x4v3.pth`：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
- `realesr-general-wdn-x4v3.pth`：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth，可选强降噪权重；和普通版放同一目录时，服务会按 `REALESRGAN_DENOISE_STRENGTH` 混合。文字保真建议 `0.2`–`0.5`，不要太高，避免笔画被抹平。

## 阿里云 OSS 输出

配置 OSS 后，`response_format=url` 的生成结果会上传到阿里云 OSS，并返回客户可访问的 OSS URL，不再返回本地 `/outputs/...` 路径。未配置 OSS 时会自动回退到本地输出目录。

在 `.env` 中配置：

```env
# ========================================
# 阿里云OSS配置
# ========================================
OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
OSS_ACCESS_KEY_ID=your_access_key_id
OSS_ACCESS_KEY_SECRET=your_access_key_secret
OSS_BUCKET=your_bucket_name
OSS_PUBLIC_BASE_URL=
OSS_OBJECT_PREFIX=images
OSS_RETENTION_DAYS=14

# ========================================
# 阿里云STS配置（用于生成临时凭证）
# ========================================
# STS角色ARN，格式: arn:ram::账号ID:role/角色名
OSS_STS_ROLE_ARN=arn:ram::1234567890:role/ComfyUIRole
# 临时凭证有效期（秒），默认3600（1小时）
OSS_STS_DURATION=3600
# 阿里云地域
ALIYUN_REGION_ID=cn-hangzhou
```

说明：

- `OSS_PUBLIC_BASE_URL` 可填 CDN 或自定义域名，例如 `https://img.example.com`；为空时默认返回 `https://bucket.endpoint/object_key`。
- `OSS_OBJECT_PREFIX` 控制对象前缀，默认上传到 `images/`。
- `OSS_RETENTION_DAYS=14` 会设置对象响应头的过期时间，真正的 14 天后自动删除需要在 OSS Bucket 中配置生命周期规则：匹配 `OSS_OBJECT_PREFIX`，文件创建后 `14` 天删除。
- `OSS_STS_ROLE_ARN`、`OSS_STS_DURATION`、`ALIYUN_REGION_ID` 目前作为 STS 配置保留；服务端上传使用 `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`。

`OSS_OBJECT_PREFIX` 示例：

| 配置 | 上传对象 Key | 适用场景 |
| --- | --- | --- |
| `OSS_OBJECT_PREFIX=images` | `images/1780xxxx-uuid.png` | 默认图片目录 |
| `OSS_OBJECT_PREFIX=iapi/images` | `iapi/images/1780xxxx-uuid.png` | 推荐生产配置，方便和其他业务隔离 |
| `OSS_OBJECT_PREFIX=` | `1780xxxx-uuid.png` | 上传到 Bucket 根目录，不推荐混合业务使用 |

建议生产环境使用 `OSS_OBJECT_PREFIX=iapi/images`，然后在 OSS 生命周期规则中匹配前缀 `iapi/images/`，设置文件创建后 `14` 天删除，避免误删 Bucket 中其他目录的文件。

## 异步任务队列

图片生成/编辑接口采用异步任务模式：

1. `POST /v1/images/generations` 或 `POST /v1/images/edits` 立即返回任务 ID；为兼容 New API，上游 HTTP 状态码返回 `200 OK`。
2. 后台 worker 从内存队列取任务，单 worker 串行执行 GPU 推理。
3. 客户端通过 `GET /v1/images/tasks/{task_id}` 获取状态和最终结果。

`.env` 队列配置：

```env
IMAGE_WORKER_COUNT=1
IMAGE_QUEUE_MAXSIZE=100
SERVICE_ROLE=all
WORKER_NAME=
TASK_QUEUE_BACKEND=memory
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_QUEUE_NAME=iapi:image_tasks
REDIS_PROCESSING_QUEUE_NAME=iapi:image_tasks:processing
REDIS_BLOCK_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=10
REDIS_SOCKET_TIMEOUT=30
REDIS_REQUEUE_STALE_ENABLED=true
REDIS_PROCESSING_TIMEOUT=90
REDIS_REQUEUE_INTERVAL=30
TASK_RUNNING_HEARTBEAT_INTERVAL=30
TASK_DB_BACKEND=sqlite
TASK_DB_PATH=data/image_tasks.sqlite3
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=iapi
MYSQL_CHARSET=utf8mb4
MYSQL_CONNECT_TIMEOUT=10
TASK_PUBLIC_BASE_URL=
```

- 单张 4090 建议保持 `IMAGE_WORKER_COUNT=1`，避免并发推理导致 OOM。
- `WORKER_NAME` 用于标识具体 Worker 节点；为空时自动使用主机名，多机部署建议显式配置为 `gpu-node-01`、`gpu-node-02` 这类稳定名称。
- 多 GPU 时可把 `IMAGE_WORKER_COUNT` 设置为 GPU 数量；当前实现会启动多个 worker，但 Diffusers pipeline 仍由同一进程管理，生产多 GPU 更推荐用“每张卡一个进程 + 不同 `DEVICE=cuda:N` + 上层负载均衡”。
- `TASK_DB_BACKEND=sqlite` 时，任务元数据会保存到 `TASK_DB_PATH` 指定的 SQLite 文件中；`TASK_DB_BACKEND=mysql` 时，会使用 `MYSQL_*` 配置连接 MySQL，并自动创建 `image_tasks` 表。
- 使用 MySQL 前需先创建数据库，例如 `MYSQL_DATABASE=iapi` 对应的库；服务只负责创建任务表，不会自动创建数据库。
- `TASK_QUEUE_BACKEND=memory` 保持单进程本地队列；`TASK_QUEUE_BACKEND=redis` 会使用远程 Redis 队列，适合 API/Worker 拆分或多节点部署。
- `SERVICE_ROLE=api` 只提供 HTTP 接口，不启动本地 worker；`SERVICE_ROLE=worker` 可配合 `python -m app.worker` 只跑 Redis worker；`SERVICE_ROLE=all` 保持 API + worker 同进程。
- `SERVICE_ROLE=api` 必须搭配 `TASK_QUEUE_BACKEND=redis`，否则 API 只入本地内存队列但没有 worker 消费。
- `REDIS_SOCKET_TIMEOUT` 必须大于 `REDIS_BLOCK_TIMEOUT`；否则 Worker 空等 Redis 队列时可能出现 `Timeout reading from Redis`。
- `REDIS_REQUEUE_STALE_ENABLED=true` 会让 Worker 定期扫描 MySQL 中 `running` 且 `updated` 超过 `REDIS_PROCESSING_TIMEOUT` 秒的任务；如果任务仍在 Redis `processing` 队列，会移回待处理队列重新执行。
- `TASK_RUNNING_HEARTBEAT_INTERVAL` 会在长任务执行期间定期刷新 MySQL `updated`，避免正常长任务被误判为超时。
- 使用本地内存队列时，服务重启后 `queued` / `running` 任务不会自动续跑；使用 Redis 队列时，任务会进入共享队列，适合 API/Worker 拆分部署。
- `TASK_PUBLIC_BASE_URL` 可配置为 FastAPI 后端公网地址；任务查询统一走 `GET /v1/images/tasks/{task_id}`。
- `queue_position` 会返回任务在待处理队列中的当前位置；任务已被 Worker 取走进入运行/处理中后会返回 `null`。
- 提交接口返回 `429 Image task queue is full` 表示队列确实达到 `IMAGE_QUEUE_MAXSIZE`；如果是 Redis/MySQL 连接失败或配置错误，会返回 `500 Failed to submit image task: ...` 并带真实错误信息。

生产推荐拆成两个启动角色，并共用同一套代码：

API 节点配置：

```env
SERVICE_ROLE=api
TASK_DB_BACKEND=mysql
TASK_QUEUE_BACKEND=redis
```

启动 API：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Worker 节点配置：

```env
SERVICE_ROLE=worker
TASK_DB_BACKEND=mysql
TASK_QUEUE_BACKEND=redis
```

启动 Worker：

```bash
python -m app.worker
```

Worker 启动后会打印当前 `SERVICE_ROLE`、`TASK_QUEUE_BACKEND`、`TASK_DB_BACKEND`、Redis 队列名等信息；如果 Redis 队列里暂时没有任务，会停留在等待状态，这是正常的。收到任务时会输出 picked/started/completed/failed 日志。

如果任务已经写入 MySQL，但 Worker 没有执行，请先确认 API 和 Worker 的 `REDIS_URL`、`REDIS_QUEUE_NAME`、`REDIS_PROCESSING_QUEUE_NAME` 完全一致。API 成功入队时会打印 `Enqueued task ... queue_size=...`；Worker 等待时会定期打印 Redis 队列长度。也可以直接在 Redis 上检查：`LLEN iapi:image_tasks` 和 `LLEN iapi:image_tasks:processing`。

API 节点不会在启动时加载模型；Worker 执行任务时会按需加载模型。多节点生产环境建议同时配置 OSS，避免输入/输出图片依赖某台机器的本地磁盘。

## 运行

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## 示例

### New API 后端测试

New API 里建议这样配置：

- Base URL：`http://127.0.0.1:8000/v1`
- 模型名：`.env` 里的 `MODEL_NAME`，默认 `flux-image-backend`
- Key：当前服务未校验密钥，可填任意非空值用于通过 New API 表单校验

模型列表测试：

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"flux-image-backend\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}"
```

流式测试也会返回兼容 SSE：

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"flux-image-backend\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}"
```

返回内容中的 `model` 字段和 `choices[0].message.content` 会使用 `.env` 中的 `MODEL_NAME`，默认 `flux-image-backend`。

### 文生图 JSON

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A cat holding a sign that says hello world\",\"size\":\"768x768\",\"num_inference_steps\":4,\"seed\":0}"
```

提交后会立即返回：

```json
{
  "id": "img-xxx",
  "object": "image.task",
  "status": "queued",
  "created": "2026-06-08T19:40:39+08:00",
  "updated": "2026-06-08T19:40:39+08:00",
  "url": "/v1/images/tasks/img-xxx"
}
```

查询任务：

```bash
curl http://127.0.0.1:8000/v1/images/tasks/img-xxx
```

任务完成后 `status=succeeded`，`result` 中包含原 OpenAI 图片响应格式。

### 图生图 JSON（图片 URL）

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A cat dressed like a wizard\",\"image\":\"https://example.com/input.png\",\"size\":\"768x768\",\"seed\":0}"
```

### 16:9 2K 海报

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A cinematic sci-fi city poster at sunset\",\"aspect_ratio\":\"16:9\",\"resolution\":\"2k\",\"seed\":0}"
```

### 9:16 4K 手机竖图

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A fantasy mobile wallpaper with glowing clouds\",\"aspect_ratio\":\"9:16\",\"resolution\":\"4k\",\"seed\":0}"
```

### 图片编辑 multipart

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits ^
  -F "prompt=A cat dressed like a wizard" ^
  -F "image=@input.png" ^
  -F "size=768x768" ^
  -F "seed=0"
```

### 图片编辑 JSON（图片 URL）

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Upscale to 4K, sharpen details, preserve original image\",\"image\":\"https://example.com/input.png\",\"aspect_ratio\":\"16:9\",\"resolution\":\"4k\",\"enhance_mode\":\"pixel\",\"seed\":0}"
```

不要黑边时传 `upscale_fit_mode=cover`，后端会保持比例并居中裁剪：

```json
{
  "prompt": "使模糊的图片修复高清，保持人物和文字一致 [enhance_mode=realesrgan aspect_ratio=16:9 resolution=4k upscale_fit_mode=cover]",
  "image": "https://example.com/input.png"
}
```

如果通过 New API 等 OpenAI 兼容网关转发，网关可能会过滤 `enhance_mode`、`aspect_ratio`、`resolution` 这类非标准字段。此时可把参数同时写进 `prompt`，后端会从 prompt 中兜底解析：

```json
{
  "prompt": "Upscale to 4K, sharpen details, preserve original image [enhance_mode=pixel aspect_ratio=16:9 resolution=4k]",
  "image": "https://example.com/input.png"
}
```

### 图片提升到 4K

本地图片提升到 4K 推荐使用 `POST /v1/images/edits`，上传原图并指定 `aspect_ratio` + `resolution=4k`。如果要保持字幕、LOGO、商品包装、海报文字不变，请显式传 `enhance_mode=pixel` 或 `enhance_mode=realesrgan`。

16:9 横版 4K，输出 `3840x2160`：

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits ^
  -F "prompt=Enhance this image to a clean high detail 4K version, preserve the original composition and subject" ^
  -F "image=@input.png" ^
  -F "aspect_ratio=16:9" ^
  -F "resolution=4k" ^
  -F "enhance_mode=pixel" ^
  -F "seed=0"
```

如果要指定精确 4K 尺寸，也可以直接传 `size`：

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits ^
  -F "prompt=Upscale to 4K, sharpen details, preserve original image" ^
  -F "image=@input.png" ^
  -F "size=3840x2160" ^
  -F "enhance_mode=pixel" ^
  -F "seed=0"
```

Real-ESRGAN 权重已配置时可改为：

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits ^
  -F "prompt=Upscale to 4K, preserve all text exactly" ^
  -F "image=@input.png" ^
  -F "size=3840x2160" ^
  -F "enhance_mode=realesrgan" ^
  -F "upscale_fit_mode=cover"
```

常用 4K 输出参数：

| 用途 | 参数 | 输出尺寸 |
| --- | --- | --- |
| 横版视频、桌面图 | `aspect_ratio=16:9`、`resolution=4k` | `3840x2160` |
| 方形图、头像 | `aspect_ratio=1:1`、`resolution=4k` | `4096x4096` |
| 手机竖图、短视频封面 | `aspect_ratio=9:16`、`resolution=4k` | `2160x3840` |
| 老照片、证件比例 | `aspect_ratio=4:3`、`resolution=4k` | `4096x3072` |

注意：`enhance_mode=flux` 是基于 FLUX 的 img2img 重绘后输出 4K 尺寸，不是 ESRGAN/Real-ESRGAN 这类纯超分；`enhance_mode=pixel` 保真但不会生成真实高清细节；如果想兼顾清晰度和保真，请配置 Real-ESRGAN 后使用 `enhance_mode=realesrgan`。

如果希望通过 FLUX 直接参考原图重绘成 4K，可以使用：

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits ^
  -F "prompt=Make this image cleaner and sharper in 4K, preserve the original composition, identity and text as much as possible" ^
  -F "image=@input.png" ^
  -F "aspect_ratio=16:9" ^
  -F "resolution=4k" ^
  -F "enhance_mode=flux" ^
  -F "flux_refine_strength=0.08" ^
  -F "seed=0"
```

`flux_refine_strength` 越低越接近原图，越高越清晰但越容易改内容。建议从 `0.05`–`0.12` 试起；如果文字变化明显，降低到 `0.03`–`0.06`。

如果希望用 Python 复刻类似 ComfyUI 工作流的 Qwen Image Edit 高清链路，可以使用：

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits ^
  -F "prompt=使模糊的图片修复高清，噪声去除，纹理自然，皮肤自然，保持人物一致性，保持文字尽量一致" ^
  -F "image=@input.png" ^
  -F "aspect_ratio=16:9" ^
  -F "resolution=4k" ^
  -F "enhance_mode=qwen_edit_realesrgan" ^
  -F "qwen_edit_strength=0.7" ^
  -F "upscale_fit_mode=cover" ^
  -F "seed=0"
```

`qwen_edit_strength` 越高，编辑修复越明显，但越可能改变原图；建议先用 `0.5`–`0.7`，人物或文字变化明显时降低。

## 响应格式

任务查询完成后的 `result` 中，`response_format=url`：

```json
{
  "created": "2026-06-08T19:40:39+08:00",
  "data": [
    {
      "url": "/outputs/example.png",
      "b64_json": null,
      "revised_prompt": "A cat holding a sign that says hello world",
      "metadata": {
        "enhance_mode": "pixel",
        "target_width": 3840,
        "target_height": 2160,
        "output_width": 3840,
        "output_height": 2160
      }
    }
  ]
}
```

配置 OSS 后，`url` 会变成类似：

```json
{
  "created": "2026-06-08T19:40:39+08:00",
  "data": [
    {
      "url": "https://your_bucket_name.oss-cn-hangzhou.aliyuncs.com/images/example.png",
      "b64_json": null,
      "revised_prompt": "A cat holding a sign that says hello world"
    }
  ]
}
```

`response_format=b64_json` 时，`data[0].b64_json` 返回 PNG base64。

## 注意事项

- 首次请求会加载模型，耗时较长且需要足够显存/内存。
- 默认 `DEVICE=auto`，优先 CUDA，其次 MPS，最后 CPU。
- 默认 `TORCH_DTYPE=bfloat16`，如果硬件不支持可改成 `float16` 或 `float32`。
- `TOKENIZERS_PARALLELISM=false` 用于关闭 tokenizer fork warning；这是提示不是错误。
- 4090 24G 跑 `FLUX.2-klein-9b-kv` 会比较贴边，默认配置使用 `768x768`、`ENABLE_CPU_OFFLOAD=true` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来优先保证稳定出图。
- 如果 `768x768` 稳定后再尝试调高 `MAX_GENERATION_PIXELS`；若仍 OOM，保持 `ENABLE_CPU_OFFLOAD=true`，并确认没有其他进程占用显存。
- 配置 OSS 后优先返回 OSS URL；未配置 OSS 时，`PUBLIC_BASE_URL` 为空会返回本地相对路径，部署到公网时建议设置为服务外部访问地址。
