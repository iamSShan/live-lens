import base64
from io import BytesIO
from PIL import Image

# from openai import OpenAI
from openai import AzureOpenAI
from typing import Dict, Any, Optional, List, Tuple
from .base_model import BaseVisionModel
from .face_recognition import FaceRecognitionManager
from .utils import create_multi_frame_person_context, preprocess_for_model
from config import Config
import logging

logger = logging.getLogger(__name__)


class OpenAIVisionModel(BaseVisionModel):
    """
    OpenAI GPT-4o model implementation
    """

    def __init__(self, face_encodings_path: str = "face_encodings.pkl"):
        super().__init__(Config.AZURE_OPENAI_MODEL_NAME)
        self.client = None
        self.face_manager = FaceRecognitionManager(face_encodings_path)

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

    def generate_description_with_faces(
        self, frames: List[Image.Image], recognize_faces: bool = True
    ) -> Tuple[str, List[List[Dict[str, Any]]]]:
        """Generate description using multiple images with face recognition"""

        images = [preprocess_for_model(img) for img in frames]
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if not all(isinstance(img, Image.Image) for img in images):
            raise ValueError("All inputs must be PIL Images")

        try:
            all_faces = []
            person_context = ""
            if recognize_faces:
                # Recognize faces in all images
                all_faces = [self.face_manager.recognize_faces(img) for img in images]
                # Create comprehensive person context
                person_context = create_multi_frame_person_context(all_faces, images)

                prompt = f"""These images are sequential frames from a webcam. {person_context}

Please provide a casual, flowing narrative that describes what's happening across all frames as if you're watching a short video clip. Focus on:
- What each person is doing (use their names when available)
- How people interact with each other if multiple people are present
- Any notable changes in position or expression for each person
- The overall scene and activity

Keep your tone conversational and natural, as if you're telling a friend what you just saw in a short video. Avoid formal analysis or frame-by-frame breakdown.
Please don't mention in which corner a person is present in frame.
"""
            else:
                prompt = """These images are sequential frames from a webcam.

Please provide a casual, flowing narrative that describes what's happening across all frames as if you're watching a short video clip. Focus on:
- What people are doing in general
- Any changes in the scene across frames
- Any interactions or activities that stand out

Keep your tone conversational and natural, as if you're telling a friend what you just saw in a short video. Avoid formal analysis or frame-by-frame breakdown.
Please don't mention in which corner a person is present in frame.
"""
            # Convert images to base64 data URLs
            image_contents = []
            for image in images:
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                image_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high",
                        },
                    }
                )

            # Build chat message with prompt and multiple images
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that describes sequences of images in a natural, conversational way. When multiple people are present, track each person's actions and interactions across frames.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + image_contents,
                },
            ]

            # Generate the response
            response = self.client.chat.completions.create(
                model=Config.AZURE_OPENAI_MODEL_NAME,
                messages=messages,
                max_tokens=250,
                temperature=0.6,
            )

            description = response.choices[0].message.content.strip()
            return description, all_faces

        except Exception as e:
            logger.error(
                f"Error generating multi-image description with faces: {str(e)}"
            )
            return f"Error: Unable to generate multi-image description - {str(e)}", []

    def generate_description(
        self, images: List[Image.Image], recognize_faces: bool = True
    ) -> str:
        """Generate multi-image description without returning face data (backward compatibility)"""
        description, _ = self.generate_description_with_faces(images, recognize_faces)
        return description

    def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
        pass
