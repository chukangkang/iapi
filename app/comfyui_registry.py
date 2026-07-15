from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model combination."""
    id: str
    name: str
    description: str = ""
    diffusion_model: Optional[str] = None
    text_encoder: Optional[str] = None
    vae: Optional[str] = None
    loras: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            diffusion_model=data.get('diffusion_model'),
            text_encoder=data.get('text_encoder'),
            vae=data.get('vae'),
            loras=data.get('loras', []),
            parameters=data.get('parameters', {}),
            metadata=data.get('metadata', {}),
        )


class ModelRegistry:
    """Registry for ComfyUI model configurations."""
    
    def __init__(self):
        self._configs: Dict[str, ModelConfig] = {}
    
    def register(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self._configs[config.id] = config
        logger.info("Registered model config: %s (%s)", config.id, config.name)
    
    def unregister(self, model_id: str) -> bool:
        """Unregister a model configuration."""
        if model_id in self._configs:
            del self._configs[model_id]
            logger.info("Unregistered model config: %s", model_id)
            return True
        return False
    
    def get(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID."""
        return self._configs.get(model_id)
    
    def list_all(self) -> List[ModelConfig]:
        """List all registered model configurations."""
        return list(self._configs.values())
    
    def has(self, model_id: str) -> bool:
        """Check if a model configuration exists."""
        return model_id in self._configs
    
    def load_from_json(self, config_path: str) -> int:
        """Load model configurations from a JSON file."""
        path = Path(config_path).expanduser().resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data.get('models', []):
            config = ModelConfig.from_dict(item)
            self.register(config)
            count += 1
        
        logger.info("Loaded %d model configs from %s", count, path)
        return count
    
    def save_to_json(self, config_path: str) -> None:
        """Save model configurations to a JSON file."""
        path = Path(config_path).expanduser().resolve()
        
        data = {
            'models': [config.to_dict() for config in self._configs.values()]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info("Saved %d model configs to %s", len(self._configs), path)
