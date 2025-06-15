import google.generativeai as genai
import base64
import io
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple
from .base_model import BaseVisionModel
from .face_recognition import FaceRecognitionManager
from .utils import create_multi_frame_person_context, preprocess_for_model
from config import Config
import logging
from PIL import (
    PngImagePlugin,
)  # https://github.com/google-gemini/deprecated-generative-ai-python/issues/178

logger = logging.getLogger(__name__)


class GeminiVisionModel(BaseVisionModel):
    """Google Gemini Vision model implementation with face recognition"""

    def __init__(self, face_encodings_path: str = "face_encodings.pkl"):
        super().__init__(Config.GEMINI_MODEL_NAME)
        self.model = None
        self.face_manager = FaceRecognitionManager(face_encodings_path)

    def load_model(self) -> None:
        """Initialize Gemini model"""
        try:
            if not Config.GEMINI_API_KEY:
                raise ValueError("Gemini API key not found in environment variables")

            genai.configure(api_key=Config.GEMINI_API_KEY)

            # Configure model
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 500,  # Increased for detailed descriptions
            }
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            self.model = genai.GenerativeModel(
                model_name=Config.GEMINI_MODEL_NAME,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            self.is_loaded = True
            logger.info("Gemini model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
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

Ensure that the description is factual and based only on what is visible in the images, avoiding any unnecessary commentary or conversational tone.
Please don't mention in which corner a person is present in frame.
"""
        else:
            prompt = """These images are sequential frames from a webcam.

Please provide a casual, flowing narrative that describes what's happening across all frames as if you're watching a short video clip. Focus on:
- What people are doing in general
- Any changes in the scene across frames
- Any interactions or activities that stand out

Ensure that the description is factual and based only on what is visible in the images, avoiding any unnecessary commentary or conversational tone.
Please don't mention in which corner a person is present in frame.
"""

        try:
            # Convert all images to RGB if needed
            clean_images = [
                img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
                for img in images
            ]

            # Gemini expects [text, image, image, image...]
            content = [prompt] + clean_images

            response = self.model.generate_content(content)

            # Check for safety blocks
            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    if candidate.finish_reason.name == "SAFETY":
                        return (
                            "Image analysis was blocked due to safety filters.",
                            all_faces,
                        )
                    elif candidate.finish_reason.name in ["RECITATION", "OTHER"]:
                        return (
                            "Unable to analyze this image due to content restrictions.",
                            all_faces,
                        )

            # Extract response
            if hasattr(response, "text") and response.text:
                return response.text.strip(), all_faces
            else:
                logger.warning("No text in Gemini multi-image response")
                return "No description could be generated from these frames.", all_faces

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Error generating multi-image description with Gemini: {error_msg}"
            )

            if "quota" in error_msg.lower():
                return "API quota exceeded. Please try again later.", all_faces
            elif "invalid" in error_msg.lower():
                return "Invalid image format or API request.", all_faces
            else:
                return f"Error analyzing multiple images: {error_msg}", all_faces

    def generate_description(
        self, images: List[Image.Image], recognize_faces: bool = True
    ) -> str:
        """Generate multi-image description without returning face data (backward compatibility)"""
        description, _ = self.generate_description_with_faces(images, recognize_faces)
        return description

    def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
        pass
