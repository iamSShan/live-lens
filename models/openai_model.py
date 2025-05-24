import base64
from io import BytesIO
from PIL import Image

# from openai import OpenAI
from openai import AzureOpenAI
from typing import Dict, Any, Optional
from .base_model import BaseVisionModel
from config import Config
import logging

logger = logging.getLogger(__name__)


class OpenAIVisionModel(BaseVisionModel):
    """
    OpenAI GPT-4V model implementation
    """

    def __init__(self):
        super().__init__("gpt-4o")
        self.client = None

    def load_model(self) -> None:
        """
        Initialize OpenAI client
        """
        try:
            if not Config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found in environment variables")

            # We can use directly OpenAI like this:
            # self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            # But here we are using AzureOpenAI:
            self.client = AzureOpenAI(
                api_key=Config.OPENAI_API_KEY,
                api_version=Config.AZURE_OPENAI_API_VERSION,  # Your specified version
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
            self.is_loaded = True
            logger.info("OpenAI client initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    def generate_description(
        self, image: Image.Image, prompt: Optional[str] = None
    ) -> str:
        """Generate description for the given image using GPT-4o"""
        if not self.is_loaded:
            print("Model was not loaded, loading it now")
            self.load_model()

        try:
            # Default prompt for scene description
            if prompt is None:
                prompt = (
                    "Describe this image in detail. Include information about people, objects, "
                    "activities, setting, and any notable features. Be specific and comprehensive."
                )

            # Convert PIL image to base64
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ]

            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=100, temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating description with OpenAI: {str(e)}")
            return f"Error: Unable to generate description - {str(e)}"

    def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
        """Fine-tune OpenAI model (placeholder - uses fine-tuning API)"""
        try:
            # Note: OpenAI fine-tuning requires preparing JSONL files and using their API
            # This is a simplified implementation

            logger.info(
                "OpenAI fine-tuning requires prepared JSONL dataset and API calls"
            )

            return {
                "status": "info",
                "message": (
                    "OpenAI fine-tuning requires dataset preparation in JSONL format "
                    "and using OpenAI's fine-tuning API. Please refer to OpenAI documentation "
                    "for detailed fine-tuning process."
                ),
            }

        except Exception as e:
            logger.error(f"Error with OpenAI fine-tuning: {str(e)}")
            return {"status": "error", "message": f"Fine-tuning setup failed: {str(e)}"}
