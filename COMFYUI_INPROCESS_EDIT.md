# ComfyUI 进程内 Qwen Image Edit

`QWEN_EDIT_BACKEND=comfyui` 时，Worker 不会启动 ComfyUI 服务，也不会调用 HTTP API。项目会把 `COMFYUI_PATH` 加入 Python 模块搜索路径，直接使用同一 Python 进程中的官方 ComfyUI 核心代码。

这不是 Diffusers 转换层。UNet、CLIP、VAE、LoRA 均保持 ComfyUI 原生 checkpoint 格式，由 `comfy.sd` 自动检测模型架构并加载。

## 对应工作流

执行链与 ComfyUI 工作流核心节点一致：

1. `UNETLoader` 等价逻辑：`comfy.sd.load_diffusion_model()`
2. `LoraLoaderModelOnly` 等价逻辑：`comfy.sd.load_lora_for_models()`
3. `CLIPLoader(type=qwen_image)` 等价逻辑：`comfy.sd.load_clip()`
4. `VAELoader` 等价逻辑：`comfy.sd.VAE()`
5. `ModelSamplingAuraFlow(shift=3.1)`
6. `TextEncodeQwenImageEditPlus`
7. `CLIPTextEncode` 负向条件
8. `EmptyLatentImage`
9. `KSampler(euler/simple)`
10. `VAEDecode`

## 配置

```env
QWEN_EDIT_BACKEND=comfyui
COMFYUI_PATH=D:/AI/ComfyUI
COMFYUI_MODELS_PATH=D:/AI/ComfyUI/models

COMFYUI_QWEN_EDIT_UNET_NAME=qwen_image_edit_2511_fp8mixed.safetensors
COMFYUI_QWEN_EDIT_UNET_WEIGHT_DTYPE=fp8_e4m3fn
COMFYUI_QWEN_EDIT_CLIP_NAME=qwen_2.5_vl_7b_fp8_scaled.safetensors
COMFYUI_QWEN_EDIT_CLIP_DEVICE=default
COMFYUI_QWEN_EDIT_VAE_NAME=qwen_image_vae.safetensors
COMFYUI_QWEN_EDIT_LORA_NAME=Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors
COMFYUI_QWEN_EDIT_MODEL_SHIFT=3.1
COMFYUI_QWEN_EDIT_SAMPLER_NAME=euler
COMFYUI_QWEN_EDIT_SCHEDULER=simple
COMFYUI_QWEN_EDIT_NEGATIVE_PROMPT=
COMFYUI_QWEN_EDIT_DEFAULT_SEED=1
```

截图工作流使用 `steps=4`、`cfg=1.0`、`denoise=1.0`、空负向文本和固定 `seed=1`。ComfyUI 后端使用独立的空负向配置，不再复用 API 中 Qwen Image 的通用负面提示词。

`qwen_edit`、`qwen_edit_realesrgan` 以及两个 unblur 模式会跳过 API 的 Prompt Enhancer。即使全局 `PROMPT_ENHANCER_ENABLED=true` 或请求传入 `prompt_enhance=true`，Edit 指令也会保持原文，避免“修复高清”被扩写成人像重绘描述。

启动日志应包含 Lightning LoRA，例如 `loras=['Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors']`。如果仍显示 `loras=[]`，说明 Worker 实际环境中的 `QWEN_EDIT_LIGHTNING_LORA_ENABLED` 为 false，或部署代码/配置未更新；此时不会与截图结果一致。

对应文件应位于：

- `models/diffusion_models/<UNET_NAME>`
- `models/text_encoders/<CLIP_NAME>`
- `models/vae/<VAE_NAME>`
- `models/loras/<LORA_NAME>`

## Python 环境

项目 Worker 与 ComfyUI 核心运行在同一个解释器中，因此必须安装该 ComfyUI checkout 对应版本的依赖。推荐在 Worker 虚拟环境中安装：

```text
pip install -r D:/AI/ComfyUI/requirements.txt
pip install -r requirements-worker.txt
```

必须保证 PyTorch/CUDA 版本与 ComfyUI 实际可运行环境一致。不能让另一个虚拟环境中的 ComfyUI 仅通过路径被导入，因为其依赖不会自动跨环境生效。

ComfyUI 的 `KSampler` 会导入 `torchsde`，而 `torchsde` 还依赖 `trampoline`。本项目已在 `requirements-worker.txt` 中显式声明两者。若旧 Worker 环境出现 `No module named 'trampoline'`，请在 Worker 实际使用的解释器中重新安装：

```text
python -m pip install -r requirements-worker.txt
# 或仅修复当前缺包：
python -m pip install "trampoline>=0.1.2" "torchsde>=0.2.6"
```

日志中的 `You need pytorch with cu130 or higher to use optimized CUDA operations` 只表示 `comfy_kitchen` 的 CUDA/Triton 优化后端被禁用。当前 `torch 2.8.0+cu128` 会回退到 eager/PyTorch 实现，不是本次导入失败原因；实际失败原因是 `trampoline` 缺失。

条件编码、采样和 VAE 解码均在 PyTorch `InferenceMode` 中执行。这样常规 VAE 解码显存不足而自动回退到 tiled decode 时，ComfyUI 对输出执行原地归一化不会触发 `Inplace update to inference tensor outside InferenceMode is not allowed`。

## 兼容边界

- 当前 Edit 路径兼容上述官方 Qwen Image Edit 核心节点及 ComfyUI 原生模型格式。
- 不执行前端 workflow JSON，也不加载工作流中的预览、保存、切片放大或第三方自定义节点；这些不属于 Edit 主采样链。
- ComfyUI 是 GPL-3.0 项目。本实现引用用户本机 checkout，没有复制其源码到本仓库；分发组合产品前仍应评估许可证义务。
- `COMFYUI_PATH` 必须与所用专用模型兼容。ComfyUI 内部 API 会随版本变化，建议固定一个已验证的 commit。