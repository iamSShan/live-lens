import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    # TrainingArguments,
    # Trainer,
)
from PIL import Image
from typing import Dict, Any, Optional
from .base_model import BaseVisionModel
from config import Config
import logging

logger = logging.getLogger(__name__)


class SmolVLMModel(BaseVisionModel):
    """SmolVLM model implementation"""

    def __init__(self):
        super().__init__(Config.SMOLVLM_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        """Load SmolVLM model and processor"""
        try:
            logger.info(f"Loading SmolVLM model: {self.model_name}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # Load model with appropriate dtype
            # self.model = AutoModelForVision2Seq.from_pretrained(
            #     self.model_name,
            #     torch_dtype=(
            #         torch.float16 if torch.cuda.is_available() else torch.float32
            #     ),
            #     device_map="auto" if torch.cuda.is_available() else None,
            # )

            # if not torch.cuda.is_available():
            #     self.model = self.model.to(self.device)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )

            self.model = self.model.to(self.device)

            self.is_loaded = True
            logger.info("SmolVLM model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading SmolVLM model: {str(e)}")
            raise

    def generate_description(
        self, image: Image.Image, prompt: Optional[str] = None
    ) -> str:
        """Generate description for the given image"""
        if not self.is_loaded:
            self.load_model()

        try:
            # Default prompt for scene description
            if prompt is None:
                # prompt = "Describe what you see in this image in detail, including people, objects, activities, and the overall scene."
                prompt = (
                    "Describe this image in detail: <image> Include information about people, objects, "
                    "activities, setting, and any notable features."
                )
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
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Extract generated text (remove input prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            print("response:", response)
            return response

        except Exception as e:
            logger.error(f"Error generating description with SmolVLM: {str(e)}")
            return f"Error: Unable to generate description - {str(e)}"

    def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
        """Fine-tune SmolVLM model"""
        pass
        # try:
        #     from datasets import load_dataset
        #     from peft import LoraConfig, get_peft_model

        #     # Load dataset
        #     dataset = load_dataset("imagefolder", data_dir=dataset_path)

        #     # Prepare LoRA configuration
        #     lora_config = LoraConfig(
        #         r=16,
        #         lora_alpha=32,
        #         lora_dropout=0.1,
        #         target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        #     )

        #     # Apply LoRA to model
        #     model_for_training = get_peft_model(self.model, lora_config)

        #     # Training arguments
        #     training_args = TrainingArguments(
        #         output_dir=f"{Config.FINE_TUNED_MODELS_DIR}/smolvlm_finetuned",
        #         num_train_epochs=Config.EPOCHS,
        #         per_device_train_batch_size=Config.BATCH_SIZE,
        #         learning_rate=Config.LEARNING_RATE,
        #         logging_steps=10,
        #         save_steps=100,
        #         eval_steps=100,
        #         warmup_steps=100,
        #         fp16=torch.cuda.is_available(),
        #     )

        #     # Initialize trainer
        #     trainer = Trainer(
        #         model=model_for_training,
        #         args=training_args,
        #         train_dataset=dataset["train"],
        #         eval_dataset=dataset.get("validation", dataset["train"]),
        #     )

        #     # Start training
        #     trainer.train()

        #     # Save model
        #     trainer.save_model()

        #     return {
        #         "status": "success",
        #         "model_path": training_args.output_dir,
        #         "message": "SmolVLM fine-tuning completed successfully",
        #     }

        # except Exception as e:
        #     logger.error(f"Error fine-tuning SmolVLM: {str(e)}")
        #     return {"status": "error", "message": f"Fine-tuning failed: {str(e)}"}
