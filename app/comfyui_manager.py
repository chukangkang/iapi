from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.comfyui_model_discovery import ModelDiscovery
from app.comfyui_registry import ModelConfig, ModelRegistry

logger = logging.getLogger(__name__)


class ComfyUIModelManager:
    """Manages ComfyUI models and provides a unified interface for model loading."""
    
    def __init__(self, models_path: Path):
        self.models_path = models_path.expanduser().resolve()
        self.discovery = ModelDiscovery(self.models_path)
        self.registry = ModelRegistry()
        self._discovered_models: Dict[str, List[Any]] = {}
    
    def discover_models(self) -> Dict[str, List[Any]]:
        """Discover all models in the ComfyUI models directory."""
        self._discovered_models = self.discovery.discover_all_models()
        return self._discovered_models
    
    def get_discovered_models(self, category: str) -> List[Any]:
        """Get discovered models by category."""
        return self._discovered_models.get(category, [])
    
    def load_registry_from_json(self, config_path: str) -> int:
        """Load model configurations from a JSON file."""
        return self.registry.load_from_json(config_path)
    
    def save_registry_to_json(self, config_path: str) -> None:
        """Save model configurations to a JSON file."""
        self.registry.save_to_json(config_path)
    
    def register_model_config(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self.registry.register(config)
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID."""
        return self.registry.get(model_id)
    
    def list_model_configs(self) -> List[ModelConfig]:
        """List all registered model configurations."""
        return self.registry.list_all()
    
    def has_model_config(self, model_id: str) -> bool:
        """Check if a model configuration exists."""
        return self.registry.has(model_id)
