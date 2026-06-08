# FLUX.2 Klein KV OpenAI Image API

一个基于 FastAPI + 🧨 Diffusers 的 OpenAI 风格图片接口服务，默认使用 `black-forest-labs/FLUX.2-klein-9b-kv` 和 `Flux2KleinKVPipeline`。

## 接口

### `POST /v1/images/generations/`

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
| `n` | 当前仅支持 `1` | `1` |

### `POST /v1/images/edits/`

使用 multipart 表单，字段与 OpenAI 图片编辑接口接近：

- `prompt`：必填
- `image`：必填，参考/待编辑图片
- `mask`：可选；当前接口会接收但 FLUX.2 Klein KV 示例未使用 mask，因此暂不参与推理
- 其他参数同 generations

## 主流比例

你可以继续用 OpenAI 风格的 `size="宽x高"`，也可以使用 `aspect_ratio` + `resolution` 请求主流尺寸：

| 场景 | `aspect_ratio` | `resolution=2k` | `resolution=4k` |
| --- | --- | --- | --- |
| 视频、海报、桌面图 | `16:9` | `2560x1440` | `3840x2160` |
| 老照片、证件、老式图片 | `4:3` | `2048x1536` | `4096x3072` |
| 方形图、头像、朋友圈 | `1:1` | `1440x1440` | `2160x2160` |
| 手机竖图、短视频封面 | `9:16` | `1440x2560` | `2160x3840` |

为了避免 4090 24G 在 2K/4K 直接推理时 OOM，服务会根据 `.env` 里的 `MAX_GENERATION_PIXELS` 先按比例生成安全尺寸，再放大到目标输出尺寸。默认 `786432` 约等于 `1024x768` 的像素预算。

## 安装

建议使用 Python 3.11+，并确保 CUDA / PyTorch 环境与显卡匹配。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果模型需要 Hugging Face 访问权限，请编辑 `.env`：

```env
HF_TOKEN=你的 HuggingFace Token
```

## 运行

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## 示例

### 文生图 JSON

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations/ ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A cat holding a sign that says hello world\",\"size\":\"768x768\",\"num_inference_steps\":4,\"seed\":0}"
```

### 图生图 JSON（图片 URL）

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations/ ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A cat dressed like a wizard\",\"image\":\"https://example.com/input.png\",\"size\":\"768x768\",\"seed\":0}"
```

### 16:9 2K 海报

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations/ ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A cinematic sci-fi city poster at sunset\",\"aspect_ratio\":\"16:9\",\"resolution\":\"2k\",\"seed\":0}"
```

### 9:16 4K 手机竖图

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations/ ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A fantasy mobile wallpaper with glowing clouds\",\"aspect_ratio\":\"9:16\",\"resolution\":\"4k\",\"seed\":0}"
```

### 图片编辑 multipart

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits/ ^
  -F "prompt=A cat dressed like a wizard" ^
  -F "image=@input.png" ^
  -F "size=768x768" ^
  -F "seed=0"
```

## 响应格式

`response_format=url`：

```json
{
  "created": 1780627200,
  "data": [
    {
      "url": "/outputs/example.png",
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
- 4090 24G 跑 `FLUX.2-klein-9b-kv` 会比较贴边，默认配置使用 `768x768`、`ENABLE_CPU_OFFLOAD=true` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来优先保证稳定出图。
- 如果 `768x768` 稳定后再尝试调高 `MAX_GENERATION_PIXELS`；若仍 OOM，保持 `ENABLE_CPU_OFFLOAD=true`，并确认没有其他进程占用显存。
- `PUBLIC_BASE_URL` 为空时返回相对路径；部署到公网时建议设置为服务外部访问地址。
