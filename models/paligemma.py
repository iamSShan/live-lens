import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from typing import Dict, Any, Optional
from .base_model import BaseVisionModel
from config import Config
import logging

logger = logging.getLogger(__name__)


class PaliGemmaVisionModel(BaseVisionModel):
    """
    PaliGemma model implementation using Hugging Face Transformers
    """

    def __init__(self):
        super().__init__("pali-gemma")
        self.model = None
        self.processor = None

        # Ensure device is defined regardless of torch version
        if hasattr(torch, "get_default_device"):
            self.device = torch.get_default_device()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_loaded = False

    def load_model(self) -> None:
        """
        Load the PaliGemma model and processor
        """
        try:
            self.processor = AutoProcessor.from_pretrained(
                "google/paligemma-3b-mix-224"
            )
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                "google/paligemma-3b-mix-224"
            ).to(self.device)
            self.is_loaded = True
            logger.info("PaliGemma model and processor loaded successfully")

        except Exception as e:
            logger.error(f"Error loading PaliGemma model: {str(e)}")
            raise

    def generate_description(
        self, image: Image.Image, prompt: Optional[str] = None
    ) -> str:
        """
        Generate description for the given image using PaliGemma
        """
        if not self.is_loaded:
            print("Model was not loaded, loading it now")
            self.load_model()

        try:
            if prompt is None:
                prompt = (
                    "Describe this image in detail. Include information about people, objects, "
                    "activities, setting, and any notable features. Be specific and comprehensive."
                )

            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
                self.device
            )

            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating description with PaliGemma: {str(e)}")
            return f"Error: Unable to generate description - {str(e)}"

    def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
        """
        Fine-tune PaliGemma model (placeholder)
        """
        try:
            logger.info(
                "PaliGemma fine-tuning support is experimental or unavailable via Hugging Face transformers"
            )

            return {
                "status": "info",
                "message": (
                    "PaliGemma fine-tuning is not currently supported via transformers. "
                    "Please refer to Google's research or use prompt-based adaptation."
                ),
            }

        except Exception as e:
            logger.error(f"Error with PaliGemma fine-tuning: {str(e)}")
            return {"status": "error", "message": f"Fine-tuning setup failed: {str(e)}"}
