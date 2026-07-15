# ComfyUI 通用模型管理系統

本項目提供了一個通用的 ComfyUI 模型管理系統，可以自動發現、加載和管理所有類型的 ComfyUI 專用模型。

## 架構概述

```
app/
├── comfyui_edit_backend.py      # 現有的 Qwen Image Edit 後端
├── comfyui_manager.py           # 通用模型管理器
├── comfyui_model_discovery.py   # 模型自動發現
├── comfyui_registry.py          # 模型註冊表
└── comfyui_inference_engine.py  # 推理引擎
```

## 核心組件

### 1. ModelDiscovery (模型發現)

自動掃描 ComfyUI 模型目錄，發現以下類型的模型：

- `diffusion_models` - Diffusion 模型（UNet/Transformer）
- `text_encoders` - 文本編碼器（CLIP/T5）
- `vae` - VAE 模型
- `loras` - LoRA 權重
- `checkpoints` - Checkpoint 文件
- `embeddings` - 嵌入文件
- `clip` - CLIP 模型

### 2. ModelRegistry (模型註冊表)

通過 JSON 配置文件定義模型組合，支持：

- 模型 ID 和名稱
- 描述
- 模型組件（Diffusion、Text Encoder、VAE、LoRA）
- 參數配置
- 元數據

### 3. ComfyUIModelManager (模型管理器)

統一的接口來管理模型：

- 發現模型
- 加載配置
- 註冊模型
- 查詢模型

## 使用指南

### 1. 配置模型

創建 `config/models.json` 文件：

```json
{
  "models": [
    {
      "id": "qwen-image-edit-2511",
      "name": "Qwen Image Edit 2511 (Lightning)",
      "description": "Qwen Image Edit 2511 with Lightning LoRA for fast editing",
      "diffusion_model": "qwen_image_edit_2511_fp8mixed.safetensors",
      "text_encoder": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
      "vae": "qwen_image_vae.safetensors",
      "loras": [
        {
          "name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
          "scale": 1.0
        }
      ],
      "parameters": {
        "shift": 3.1,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "cfg": 1.0,
        "negative_prompt": ""
      },
      "metadata": {
        "type": "image_edit",
        "max_images": 3,
        "supports_mask": true
      }
    }
  ]
}
```

### 2. 使用模型管理器

```python
from pathlib import Path
from app.comfyui_manager import ComfyUIModelManager

# 初始化模型管理器
manager = ComfyUIModelManager(Path("D:/AI/ComfyUI/models"))

# 發現所有模型
models = manager.discover_models()
print(f"Discovered {len(models)} model categories")

# 加載模型配置
manager.load_registry_from_json("config/models.json")

# 查詢模型配置
config = manager.get_model_config("qwen-image-edit-2511")
print(f"Model: {config.name}")

# 列出所有註冊的模型
all_configs = manager.list_model_configs()
for cfg in all_configs:
    print(f"- {cfg.id}: {cfg.name}")
```

### 3. 添加新模型

只需在 `config/models.json` 中添加新的模型配置：

```json
{
  "id": "stable-diffusion-xl",
  "name": "Stable Diffusion XL",
  "description": "SDXL base model for text-to-image",
  "diffusion_model": "sdxl_base.safetensors",
  "text_encoder": "clip_sdxl.safetensors",
  "vae": "vae_sdxl.safetensors",
  "loras": [],
  "parameters": {
    "shift": 1.0,
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 30,
    "cfg": 7.0,
    "negative_prompt": "blurry, low resolution"
  },
  "metadata": {
    "type": "text_to_image",
    "max_images": 1,
    "supports_mask": false
  }
}
```

## 支持的模型類型

這個系統設計為支持所有 ComfyUI 專用模型：

- **Qwen Image 系列**
  - Qwen Image 2512 (Text-to-Image)
  - Qwen Image Edit 2511 (Image Editing)
  
- **Stable Diffusion 系列**
  - SD 1.5
  - SD 2.0
  - SDXL
  - SDXL Turbo
  
- **其他 ComfyUI 兼容模型**
  - Stable Cascade
  - AuraFlow
  - 任何使用 ComfyUI 格式的模型

## 擴展指南

### 添加新的模型類型

1. 在 `config/models.json` 中添加模型配置
2. 確保模型文件位於正確的 ComfyUI 模型目錄中
3. 使用 `ModelDiscovery` 驗證模型是否被正確發現

### 自定義參數

每個模型都可以有自定義參數：

```json
"parameters": {
  "shift": 3.1,
  "sampler": "euler",
  "scheduler": "simple",
  "steps": 4,
  "cfg": 1.0,
  "negative_prompt": "",
  "custom_param": "value"
}
```

## 注意事項

- 模型文件必須位於 ComfyUI 的模型目錄中
- 模型文件名必須與配置文件中的名稱匹配
- 確保 ComfyUI 依賴已安裝

## 後續開發

未來的改進可能包括：

- 支持 YAML 配置文件
- 模型版本管理
- 模型熱加載
- 模型緩存策略
- 自動模型下載
