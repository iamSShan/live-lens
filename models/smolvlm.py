import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)
from PIL import Image
from typing import Dict, Any, Optional, List
from .base_model import BaseVisionModel

# from face_recognition import recognize_faces_and_generate_prompt
from config import Config
import logging
from transformers import AutoModelForVision2Seq
from .utils import (
    apply_transformers_fix,
    preprocess_for_model,
    select_best_frame,
    create_single_image_person_context,
)
from .face_recognition import FaceRecognitionManager


logger = logging.getLogger(__name__)


class SmolVLMModel(BaseVisionModel):
    """SmolVLM model implementation"""

    def __init__(self, face_encodings_path: str = "face_encodings.pkl"):
        super().__init__(Config.SMOLVLM_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_manager = FaceRecognitionManager(face_encodings_path)

    def load_model(self) -> None:
        """Load SmolVLM model and processor"""
        try:
            logger.info(f"Loading SmolVLM model: {self.model_name}")

            # Apply patch before model loading
            apply_transformers_fix()

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, token=Config.HUGGINGFACE_TOKEN
            )

            # Load model with appropriate dtype
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                token=Config.HUGGINGFACE_TOKEN,
            )

            self.model = self.model.to(self.device)
            self.is_loaded = True
            logger.info(f"SmolVLM model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading SmolVLM model: {str(e)}")
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

            if person_context:
                prompt = f"<image> {person_context} What do you see in this image? Please use names of identified people."
            else:
                prompt = "<image> What do you see in this image?"
            # Prepare inputs
            inputs = self.processor(
                text=[prompt],  # Wrap in list
                images=[image],  # Wrap in list
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    min_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    repetition_penalty=1.2,  # Discourage repetition
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Extract generated text (remove input prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()

            # Post-processing to remove repetitive instructions
            lines = response.split("\n")
            filtered_lines = []
            for line in lines:
                if not any(
                    phrase in line.lower()
                    for phrase in ["describe the", "describe this", "describe image"]
                ):
                    filtered_lines.append(line)

            # If filtering removed everything, return original response
            clean_response = (
                "\n".join(filtered_lines).strip() if filtered_lines else response
            )
            # Final check to ensure we have content
            if not clean_response or len(clean_response) < 20:
                return response  # Return original if cleaned version is too short

            # print("response:", clean_response)
            return clean_response

        except Exception as e:
            logger.error(f"Error generating description with SmolVLM: {str(e)}")
            return f"Error: Unable to generate description - {str(e)}"

    def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
        """Fine-tune SmolVLM model"""
        pass
