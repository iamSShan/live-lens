from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict, Any, List, Optional


class BaseVisionModel(ABC):
    """Abstract base class for vision models"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor"""
        pass

    @abstractmethod
    def generate_description(
        self, image: Image.Image, prompt: Optional[str] = None
    ) -> str:
        """Generate description for the given image"""
        pass

    @abstractmethod
    def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
        """Fine-tune the model on custom dataset"""
        pass

    def unload_model(self) -> None:
        """Unload model to free memory"""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_loaded = False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "supports_fine_tuning": True,
        }
