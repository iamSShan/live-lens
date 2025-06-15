import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import Dict, Any, Optional, List
from .base_model import BaseVisionModel
from config import Config
import logging
from .utils import (
    apply_transformers_fix,
    preprocess_for_model,
    select_best_frame,
    create_single_image_person_context,
)
from .face_recognition import FaceRecognitionManager


logger = logging.getLogger(__name__)


class QwenVLModel(BaseVisionModel):
    """Qwen2-VL model implementation"""

    def __init__(self, face_encodings_path: str = "face_encodings.pkl"):
        super().__init__(Config.QWEN_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_manager = FaceRecognitionManager(face_encodings_path)

    def load_model(self) -> None:
        """Load Qwen2-VL model and processor"""
        try:
            logger.info(f"Loading Qwen2-VL model: {self.model_name}")

            # Apply patch before model loading
            apply_transformers_fix()

            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                token=Config.HUGGINGFACE_TOKEN,
            )

            self.model = self.model.to(self.device)

            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, token=Config.HUGGINGFACE_TOKEN
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=Config.HUGGINGFACE_TOKEN
            )

            self.is_loaded = True
            logger.info(f"Qwen2-VL model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading Qwen2-VL model: {str(e)}")
            raise

    def generate_description(
        self, images: List[Image.Image], recognize_faces: bool = True
    ) -> str:
        """Generate description for the given image"""
        if not self.is_loaded:
            self.load_model()

        try:
            # Filter out best image out of all frames
            best_image = select_best_frame(images)
            image = preprocess_for_model(best_image)

            # Handle face recognition
            person_context = ""
            if recognize_faces:
                # Use your existing face recognition code
                faces = self.face_manager.recognize_faces(image)

                if faces:
                    # Create context about recognized people
                    person_context = create_single_image_person_context(faces, image)

            print("person_context:", person_context)
            # Default prompt for scene description
            if person_context:
                # Use a system message to provide context about recognized people
                system_message = f"Important context: {person_context}"

                prompt = (
                    "Describe this image in a natural, conversational tone. "
                    "Make sure to refer to the people by their names as provided in the context. "
                    "Focus on what the identified people are doing, their expressions, and how they "
                    "interact with the environment and each other."
                )
            else:
                system_message = (
                    "You are a helpful assistant that describes images in detail."
                )
                prompt = (
                    "Describe this image in natural, conversational tone. Include information about people, objects, "
                    "activities, setting, and any notable features."
                )

            # Prepare conversation format with system message
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    min_new_tokens=50,  # Ensure a minimum length
                    temperature=0.3,
                    do_sample=True,
                    repetition_penalty=1.3,  # Stronger penalty for repetition
                    top_p=0.9,  # Focus on more likely tokens
                    no_repeat_ngram_size=3,  # Avoid repeating phrases
                )

            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating description with Qwen2-VL: {str(e)}")
            return f"Error: Unable to generate description - {str(e)}"

    def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
        """Fine-tune Qwen2-VL model"""
        pass
