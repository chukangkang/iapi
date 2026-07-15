"""
ComfyUI 模型管理系統使用示例

這個腳本展示了如何使用 ComfyUIModelManager 來發現、加載和管理模型。
"""

from pathlib import Path
from app.comfyui_manager import ComfyUIModelManager


def main():
    # 1. 初始化模型管理器
    # 將這裡的路徑改為您的 ComfyUI 模型目錄
    models_path = Path("D:/AI/ComfyUI/models")
    
    if not models_path.exists():
        print(f"模型目錄不存在: {models_path}")
        print("請將 'D:/AI/ComfyUI/models' 改為您的實際路徑")
        return
    
    manager = ComfyUIModelManager(models_path)
    
    # 2. 發現所有模型
    print("=" * 60)
    print("步驟 1: 發現模型")
    print("=" * 60)
    
    models = manager.discover_models()
    for category, model_list in models.items():
        print(f"\n{category}: {len(model_list)} 個模型")
        for model in model_list[:3]:  # 只顯示前3個
            print(f"  - {model.name}")
    
    # 3. 加載模型配置
    print("\n" + "=" * 60)
    print("步驟 2: 加載模型配置")
    print("=" * 60)
    
    config_path = Path(__file__).parent / "config" / "models.json"
    if config_path.exists():
        count = manager.load_registry_from_json(str(config_path))
        print(f"從 {config_path} 加載了 {count} 個模型配置")
    
    # 4. 查詢模型配置
    print("\n" + "=" * 60)
    print("步驟 3: 查詢模型配置")
    print("=" * 60)
    
    config = manager.get_model_config("qwen-image-edit-2511")
    if config:
        print(f"\n模型 ID: {config.id}")
        print(f"模型名稱: {config.name}")
        print(f"描述: {config.description}")
        print(f"Diffusion Model: {config.diffusion_model}")
        print(f"Text Encoder: {config.text_encoder}")
        print(f"VAE: {config.vae}")
        print(f"LoRAs: {[lora['name'] for lora in config.loras]}")
    
    # 5. 列出所有註冊的模型
    print("\n" + "=" * 60)
    print("步驟 4: 列出所有註冊的模型")
    print("=" * 60)
    
    all_configs = manager.list_model_configs()
    for cfg in all_configs:
        print(f"- {cfg.id}: {cfg.name}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
