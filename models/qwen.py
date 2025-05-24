import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import Dict, Any, Optional
from .base_model import BaseVisionModel
from config import Config
import logging

logger = logging.getLogger(__name__)


class QwenVLModel(BaseVisionModel):
    """Qwen2-VL model implementation"""

    def __init__(self):
        super().__init__(Config.QWEN_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        """Load Qwen2-VL model and processor"""
        try:
            logger.info(f"Loading Qwen2-VL model: {self.model_name}")

            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )

            self.model = self.model.to(self.device)

            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

            self.is_loaded = True
            logger.info("Qwen2-VL model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Qwen2-VL model: {str(e)}")
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
                prompt = "Describe this image in detail, including all people, objects, activities, and the setting."

            # Prepare conversation format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
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
                    **inputs, max_new_tokens=200, temperature=0.7, do_sample=True
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
        # try:
        #     from datasets import load_dataset
        #     from peft import LoraConfig, get_peft_model
        #     from transformers import TrainingArguments, Trainer

        #     # Load dataset
        #     dataset = load_dataset("imagefolder", data_dir=dataset_path)

        #     # Prepare LoRA configuration
        #     lora_config = LoraConfig(
        #         r=64,
        #         lora_alpha=128,
        #         lora_dropout=0.05,
        #         target_modules=[
        #             "q_proj",
        #             "k_proj",
        #             "v_proj",
        #             "o_proj",
        #             "gate_proj",
        #             "up_proj",
        #             "down_proj",
        #         ],
        #     )

        #     # Apply LoRA to model
        #     model_for_training = get_peft_model(self.model, lora_config)

        #     # Training arguments
        #     training_args = TrainingArguments(
        #         output_dir=f"{Config.FINE_TUNED_MODELS_DIR}/qwen_finetuned",
        #         num_train_epochs=Config.EPOCHS,
        #         per_device_train_batch_size=Config.BATCH_SIZE,
        #         learning_rate=Config.LEARNING_RATE,
        #         logging_steps=10,
        #         save_steps=100,
        #         eval_steps=100,
        #         warmup_steps=100,
        #         fp16=torch.cuda.is_available(),
        #         dataloader_pin_memory=False,
        #     )

        #     # Initialize trainer
        #     trainer = Trainer(
        #         model=model_for_training,
        #         args=training_args,
        #         train_dataset=dataset["train"],
        #         eval_dataset=dataset.get("validation", dataset["train"]),
        #         tokenizer=self.tokenizer,
        #     )

        #     # Start training
        #     trainer.train()

        #     # Save model
        #     trainer.save_model()

        #     return {
        #         "status": "success",
        #         "model_path": training_args.output_dir,
        #         "message": "Qwen2-VL fine-tuning completed successfully",
        #     }

        # except Exception as e:
        #     logger.error(f"Error fine-tuning Qwen2-VL: {str(e)}")
        #     return {"status": "error", "message": f"Fine-tuning failed: {str(e)}"}
