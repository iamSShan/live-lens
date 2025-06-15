import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from typing import Dict, Any, Optional, List
from .base_model import BaseVisionModel
from .utils import (
    apply_transformers_fix,
    preprocess_for_model,
    select_best_frame,
    create_single_image_person_context,
)
from .face_recognition import FaceRecognitionManager
from config import Config
import logging


logger = logging.getLogger(__name__)


class PaliGemmaVisionModel(BaseVisionModel):
    """
    PaliGemma model implementation using Hugging Face Transformers
    """

    def __init__(self, face_encodings_path: str = "face_encodings.pkl"):
        super().__init__(Config.PALIGEMMA_MODEL_NAME)
        self.model = None
        self.processor = None
        self.face_manager = FaceRecognitionManager(face_encodings_path)

        # Ensure device is defined regardless of torch version
        # if hasattr(torch, "get_default_device"):
        #     self.device = torch.get_default_device()
        # else:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        """
        Load the PaliGemma model and processor
        """
        try:
            logger.info(f"Loading Paligemma model: {self.model_name}")

            # Apply patch before model loading
            apply_transformers_fix()

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, token=Config.HUGGINGFACE_TOKEN
            )
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_name, token=Config.HUGGINGFACE_TOKEN
            ).to(self.device)
            self.is_loaded = True
            logger.info(
                f"PaliGemma model and processor loaded on {self.device} successfully"
            )

        except Exception as e:
            logger.error(f"Error loading PaliGemma model: {str(e)}")
            raise

    # def generate_description(
    #     self, images: List[Image.Image], prompt: Optional[str] = None
    # ) -> str:
    #     """
    #     Generate description for the given image using PaliGemma
    #     """
    #     if not self.is_loaded:
    #         print("Model was not loaded, loading it now")
    #         self.load_model()

    #     try:
    #         # Filter out best image out of all frames
    #         best_image = select_best_frame(images)
    #         image = preprocess_for_model(best_image)

    #         if prompt is None:
    #             prompt = (
    #                 "<image> Describe this image in detail. Include information about people, objects, "
    #                 "activities, setting, and any notable features. Be specific and comprehensive."
    #             )

    #         inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
    #             self.device
    #         )

    #         generated_ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=True,
    #                             top_p=0.9,temperature=0.7)
    #         generated_text = self.processor.batch_decode(
    #             generated_ids, skip_special_tokens=True
    #         )[0]
    #         # Remove prompt text if it's echoed at the start
    #         if generated_text.startswith(prompt[7:]):
    #             generated_text = generated_text[len(prompt[7:]):].strip()
    #         return generated_text.strip()

    #     except Exception as e:
    #         logger.error(f"Error generating description with PaliGemma: {str(e)}")
    #         return f"Error: Unable to generate description - {str(e)}"

    def generate_description(
        self, images: List[Image.Image], recognize_faces: bool = True
    ) -> str:
        """
        Generate description for the given image using PaliGemma
        """
        if not self.is_loaded:
            print("Model was not loaded, loading it now")
            self.load_model()

        # try:
        # Filter out best image out of all frames
        best_image = select_best_frame(images)
        image = preprocess_for_model(best_image)
        # Handle face recognition
        person_context = ""
        print("recognize_faces;", recognize_faces)
        if recognize_faces:
            # Use your existing face recognition code
            faces = self.face_manager.recognize_faces(image)

            if faces:
                # Create context about recognized people
                # We'll use a simplified version since PaliGemma is used with a single image
                person_context = create_single_image_person_context(faces, image)

        print("facessss", person_context)
        if person_context:
            prompt = (
                "<image> "
                f"{person_context} "
                "Describe what you see in this image in a natural, conversational tone. "
                "Focus on what the identified people are doing, their expressions, and interactions."
            )
        else:
            prompt = (
                "<image> "
                "Describe this image in detail. Include information about people, objects, "
                "activities, setting, and any notable features."
            )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.5,
            repetition_penalty=1.2,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        # Remove prompt text if it's echoed at the start
        if generated_text.startswith(prompt[7:]):
            generated_text = generated_text[len(prompt[7:]) :].strip()
        return generated_text.strip()

        # except Exception as e:
        #     logger.error(f"Error generating description with PaliGemma: {str(e)}")
        #     return f"Error: Unable to generate description - {str(e)}"
