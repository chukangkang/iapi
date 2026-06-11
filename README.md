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
| `image` | 可选参考图；存在时走 i2i/KV cache 流程 | `null` |
| `size` | OpenAI 风格尺寸，如 `768x768`、`1024x1024` | `768x768` |
| `aspect_ratio` | 主流比例预设：`16:9`、`4:3`、`1:1`、`9:16` | `null` |
| `resolution` | 搭配 `aspect_ratio` 使用：`2k` 或 `4k` | `2k` |
| `width` / `height` | 显式宽高，会覆盖 `size` 对应维度 | `.env` 默认值 |
| `num_inference_steps` | 推理步数 | `4` |
| `seed` | 随机种子 | `null` |
| `response_format` | `url` 或 `b64_json` | `url` |
| `enhance_mode` | 高清/保真模式：`flux`、`pixel`、`realesrgan`、`realesrgan_flux` | `.env` 默认值 |
| `flux_refine_strength` | `realesrgan_flux` 时传给 FLUX 的低重绘强度；pipeline 不支持时会自动忽略 | `0.08` |
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
| 方形图、头像、朋友圈 | `1:1` | `1440x1440` | `2160x2160` |
| 手机竖图、短视频封面 | `9:16` | `1440x2560` | `2160x3840` |

为了避免 4090 24G 在 2K/4K 直接推理时 OOM，`enhance_mode=flux` 会根据 `.env` 里的 `MAX_GENERATION_PIXELS` 先按比例生成安全尺寸，再放大到目标输出尺寸。默认 `786432` 约等于 `1024x768` 的像素预算。

如果目标是视频高清、海报高清、字幕/商品图文字保真，建议不要用默认 FLUX 重绘，而是使用 `enhance_mode=pixel` 或 `enhance_mode=realesrgan`。

## 安装

建议使用 Python 3.11+，并确保 CUDA / PyTorch 环境与显卡匹配。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果模型需要 Hugging Face 访问权限，请编辑 `.env`：

```env
MODEL_NAME=flux-image-backend
HF_TOKEN=你的 HuggingFace Token
TOKENIZERS_PARALLELISM=false
```

`MODEL_NAME` 是专门返回给 New API / OpenAI 兼容测试的模型名，默认 `flux-image-backend`；实际加载的 Hugging Face 模型仍由 `MODEL_PATH` 控制。

`TOKENIZERS_PARALLELISM=false` 用于关闭 Hugging Face `tokenizers` 在多进程/fork 场景下的并行 warning，不影响图片生成结果。

### 高清保真模式

`.env` 可配置：

```env
DEFAULT_ENHANCE_MODE=flux
FLUX_REFINE_STRENGTH=0.08
REALESRGAN_MODEL_PATH=
REALESRGAN_TILE=512
```

模式说明：

| 模式 | 说明 | 文字一致性 |
| --- | --- | --- |
| `flux` | 当前默认：FLUX 图生图/文生图，适合创作和重绘 | 可能改变文字 |
| `pixel` | 只用 Lanczos 像素放大，不进扩散模型 | 最稳定 |
| `realesrgan` | 用 Real-ESRGAN 超分，失败前需安装依赖并配置权重路径 | 高 |
| `realesrgan_flux` | 先 Real-ESRGAN，再尝试 FLUX 极低强度细节修复 | 仍可能轻微改变 |

严格要求“字体、文字 100% 不变”时，优先 `pixel`；安装并配置 Real-ESRGAN 权重后可试 `realesrgan`。`realesrgan_flux` 会再次进入 FLUX，虽然默认强度很低，但扩散模型没有像素级一致性保证。

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
TASK_DB_PATH=data/image_tasks.sqlite3
TASK_PUBLIC_BASE_URL=
```

- 单张 4090 建议保持 `IMAGE_WORKER_COUNT=1`，避免并发推理导致 OOM。
- 多 GPU 时可把 `IMAGE_WORKER_COUNT` 设置为 GPU 数量；当前实现会启动多个 worker，但 Diffusers pipeline 仍由同一进程管理，生产多 GPU 更推荐用“每张卡一个进程 + 不同 `DEVICE=cuda:N` + 上层负载均衡”。
- 任务元数据会保存到 `TASK_DB_PATH` 指定的 SQLite 文件中，服务重启后仍可查询已保存的任务状态和已完成结果。
- 当前待执行队列仍在内存中，服务重启时 `queued` / `running` 任务不会自动续跑；如需生产级可靠队列，可接 Redis/RQ/Celery。
- `TASK_PUBLIC_BASE_URL` 可配置为 FastAPI 后端公网地址；任务查询统一走 `GET /v1/images/tasks/{task_id}`。

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
  "created": "2026-06-08T11:40:39Z",
  "updated": "2026-06-08T11:40:39Z",
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
  -F "enhance_mode=realesrgan"
```

常用 4K 输出参数：

| 用途 | 参数 | 输出尺寸 |
| --- | --- | --- |
| 横版视频、桌面图 | `aspect_ratio=16:9`、`resolution=4k` | `3840x2160` |
| 方形图、头像 | `aspect_ratio=1:1`、`resolution=4k` | `2160x2160` |
| 手机竖图、短视频封面 | `aspect_ratio=9:16`、`resolution=4k` | `2160x3840` |
| 老照片、证件比例 | `aspect_ratio=4:3`、`resolution=4k` | `4096x3072` |

注意：`enhance_mode=flux` 是基于 FLUX 的 img2img 重绘后输出 4K 尺寸，不是 ESRGAN/Real-ESRGAN 这类纯超分；如果想尽量保持原图，请使用 `enhance_mode=pixel` 或 `enhance_mode=realesrgan`。

## 响应格式

任务查询完成后的 `result` 中，`response_format=url`：

```json
{
  "created": "2026-06-08T11:40:39Z",
  "data": [
    {
      "url": "/outputs/example.png",
      "b64_json": null,
      "revised_prompt": "A cat holding a sign that says hello world"
    }
  ]
}
```

配置 OSS 后，`url` 会变成类似：

```json
{
  "created": "2026-06-08T11:40:39Z",
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
