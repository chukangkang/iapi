from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model file."""
    name: str
    type_: str  # diffusion_models, text_encoders, vae, loras, checkpoints
    path: Path
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRegistryEntry:
    """A registered model combination."""
    id: str
    name: str
    description: str = ""
    diffusion_model: Optional[str] = None
    text_encoder: Optional[str] = None
    vae: Optional[str] = None
    loras: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelDiscovery:
    """Discovers models in ComfyUI model directories."""
    
    def __init__(self, models_path: Path):
        self.models_path = models_path.expanduser().resolve()
    
    def discover_models(self, category: str) -> List[ModelInfo]:
        """Discover models in a specific category directory."""
        category_dir = self.models_path / category
        models = []
        
        if not category_dir.exists():
            logger.warning("Model directory not found: %s", category_dir)
            return models
        
        for file_path in category_dir.rglob('*'):
            if file_path.is_file() and _is_model_file(file_path):
                models.append(ModelInfo(
                    name=file_path.name,
                    type_=category,
                    path=file_path,
                    size_bytes=file_path.stat().st_size,
                ))
        
        logger.info("Discovered %d models in %s", len(models), category)
        return models
    
    def discover_all_models(self) -> Dict[str, List[ModelInfo]]:
        """Discover all models across all categories."""
        categories = [
            'diffusion_models',
            'text_encoders',
            'vae',
            'loras',
            'checkpoints',
            'embeddings',
            'clip',
        ]
        
        all_models = {}
        for category in categories:
            models = self.discover_models(category)
            if models:
                all_models[category] = models
        
        return all_models


def _is_model_file(path: Path) -> bool:
    """Check if a file is a model file based on extension."""
    model_extensions = {'.safetensors', '.pt', '.pth', '.bin', '.ckpt'}
    return path.suffix.lower() in model_extensions
