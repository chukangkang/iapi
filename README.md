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
| `size` | OpenAI 风格尺寸，如 `1024x1024` | `1024x1024` |
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
  -d "{\"prompt\":\"A cat holding a sign that says hello world\",\"size\":\"1024x1024\",\"num_inference_steps\":4,\"seed\":0}"
```

### 图生图 JSON（图片 URL）

```bash
curl -X POST http://127.0.0.1:8000/v1/images/generations/ ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A cat dressed like a wizard\",\"image\":\"https://example.com/input.png\",\"size\":\"1024x1024\",\"seed\":0}"
```

### 图片编辑 multipart

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits/ ^
  -F "prompt=A cat dressed like a wizard" ^
  -F "image=@input.png" ^
  -F "size=1024x1024" ^
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
- `ENABLE_CPU_OFFLOAD=true` 可降低 CUDA 显存压力，但速度会下降。
- `PUBLIC_BASE_URL` 为空时返回相对路径；部署到公网时建议设置为服务外部访问地址。
