# These are some extra functions which I created but are not currently used
# Keeping here for reference


################### Gemini ###################


def generate_description_with_faces(
    self, image: Image.Image, prompt: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate description with face recognition"""
    if not self.is_loaded:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    # Recognize faces first
    recognized_faces = self.face_manager.recognize_faces(image)

    # Create person context for the prompt
    person_context = self._create_person_context(recognized_faces)

    # Enhanced prompt with face information
    if prompt is None:
        prompt = f"""Describe what you see in this webcam image. {person_context}

Include:
- What each identified person is doing (use their names when provided)
- Objects, furniture, or items visible
- The setting/environment
- Any activities or interactions between people
- Body language and expressions

Keep it detailed but concise."""

    try:
        # Validate image input
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")

        # Ensure image is in compatible format
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        # Prepare content for Gemini API
        content = [prompt, image]

        # Generate response
        response = self.model.generate_content(content)

        # Handle safety blocks
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                if candidate.finish_reason.name == "SAFETY":
                    return (
                        "Image analysis was blocked due to safety filters.",
                        recognized_faces,
                    )
                elif candidate.finish_reason.name in ["RECITATION", "OTHER"]:
                    return (
                        "Unable to analyze this image due to content restrictions.",
                        recognized_faces,
                    )

        # Extract text response
        if hasattr(response, "text") and response.text:
            return response.text.strip(), recognized_faces
        else:
            logger.warning("No text in Gemini response")
            return "No description could be generated for this image.", recognized_faces

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error generating description with Gemini: {error_msg}")

        # Handle specific error types
        if "quota" in error_msg.lower():
            return "API quota exceeded. Please try again later.", recognized_faces
        elif "invalid" in error_msg.lower():
            return "Invalid image format or API request.", recognized_faces
        else:
            return f"Error analyzing image: {error_msg}", recognized_faces


def _create_person_context(self, faces: List[Dict[str, Any]]) -> str:
    """Create context string for identified people"""
    if not faces:
        return ""

    identified_people = [f["name"] for f in faces if f["name"] != "Unknown"]
    unknown_count = len([f for f in faces if f["name"] == "Unknown"])

    context_parts = []
    if identified_people:
        context_parts.append(
            f"Identified people in the image: {', '.join(identified_people)}"
        )
    if unknown_count > 0:
        context_parts.append(f"There are also {unknown_count} unidentified person(s)")

    return " ".join(context_parts) + "." if context_parts else ""


def _create_multi_frame_person_context(
    self, all_faces: List[List[Dict[str, Any]]]
) -> str:
    """Create context string for people across multiple frames"""
    all_identified = set()
    total_unknown = 0

    for frame_faces in all_faces:
        for face in frame_faces:
            if face["name"] != "Unknown":
                all_identified.add(face["name"])
            else:
                total_unknown += 1

    context_parts = []
    if all_identified:
        context_parts.append(
            f"People identified across frames: {', '.join(sorted(all_identified))}"
        )
    if total_unknown > 0:
        context_parts.append(f"There are also unidentified person(s) in some frames")

    return " ".join(context_parts) + "." if context_parts else ""


# Keep the original methods for backward compatibility
def generate_description(self, image: Image.Image, prompt: Optional[str] = None) -> str:
    """Generate description without face recognition (backward compatibility)"""
    description, _ = self.generate_description_with_faces(image, prompt)
    return description

    ################### GPT-4o ###################

    # def generate_description_multi(
    #     self, images: List[Image.Image], prompt: Optional[str] = None
    # ) -> str:
    #     """Generate description from multiple images using GPT-4o"""

    #     if not self.is_loaded:
    #         print("Model was not loaded, loading it now")
    #         self.load_model()

    #     try:
    #         # Default prompt
    #         if prompt is None:
    #             prompt = (
    #                 "Describe what is happening across these images in detail. "
    #                 "Include people, actions, objects, setting, and any changes between frames."
    #             )

    #         # Convert images to base64 data URLs
    #         image_contents = []
    #         for image in images:
    #             if image.mode not in ('RGB', 'RGBA'):
    #                 image = image.convert("RGB")
    #             buffer = BytesIO()
    #             image.save(buffer, format="JPEG")
    #             image_base64 = base64.b64encode(buffer.getvalue()).decode()
    #             image_contents.append({
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url": f"data:image/jpeg;base64,{image_base64}",
    #                     "detail": "high",
    #                 },
    #             })

    #         # Build chat message with prompt and multiple images
    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": [{"type": "text", "text": prompt}] + image_contents,
    #             }
    #         ]
    #         # Generate the response
    #         response = self.client.chat.completions.create(
    #             model="gpt-4o",
    #             messages=messages,
    #             max_tokens=250,
    #             temperature=0.7
    #         )

    #         return response.choices[0].message.content.strip()

    #     except Exception as e:
    #         logger.error(f"Error generating multi-image description with GPT-4o: {str(e)}")
    #         return f"Error: Unable to generate multi-image description - {str(e)}"

    def _create_person_context(self, faces: List[Dict[str, Any]]) -> str:
        """Create context string for identified people in a single frame"""
        if not faces:
            return ""

        identified = [face["name"] for face in faces if face["name"] != "Unknown"]
        unknown_count = sum(1 for face in faces if face["name"] == "Unknown")

        context_parts = []
        if identified:
            context_parts.append(f"People identified: {', '.join(identified)}")
        if unknown_count > 0:
            context_parts.append(
                f"There {'is' if unknown_count == 1 else 'are'} also {unknown_count} unidentified person{'s' if unknown_count > 1 else ''}"
            )

        return " ".join(context_parts) + "." if context_parts else ""


# Keep the original methods for backward compatibility
def generate_description(self, image: Image.Image, prompt: Optional[str] = None) -> str:
    """Generate description without face recognition (backward compatibility)"""
    description, _ = self.generate_description_with_faces(image, prompt)
    return description


################### SmolVLM ###################


def generate_description_multi(
    self, images: List[Image.Image], prompt: Optional[str] = None
) -> str:
    """Generate description for multiple images using SmolVLM-style model"""
    if not self.is_loaded:
        self.load_model()

    try:
        # Default prompt for scene description
        if prompt is None:
            # prompt = (
            #     "Describe the images in detail: "
            #     + " ".join(["<image>"] * len(images))
            #     + " Include information about people, objects, activities, setting, and any notable features."
            # )
            prompt = (
                "Please provide a detailed and coherent description of the following images. "
                "Include people, actions, objects, the environment, and any changes or interactions between frames."
                + " "
                + " ".join(["<image>"] * len(images))
            )
        # Resize images to a consistent size if needed
        images = [img.resize((224, 224)) for img in images]  # Example resizing step

        # Prepare inputs with list of images
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt text if present
        if prompt in response:
            response = response.split(prompt)[-1].strip()

        print("response:", response)
        return response

    except Exception as e:
        logger.error(f"Error generating multi-image description with SmolVLM: {str(e)}")
        return f"Error: Unable to generate multi-image description - {str(e)}"


################### Qwen ###################


def generate_description_multi(
    self, images: List[Image.Image], prompt: Optional[str] = None
) -> str:
    """Generate a description from multiple images using Qwen"""

    print(len(images))
    if not self.is_loaded:
        self.load_model()

    try:
        # Default prompt
        if prompt is None:
            prompt = "Describe the scene using all the images. Include people, objects, activities, and setting."

        # Compose messages: multiple image blocks followed by prompt
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": prompt}]
                ),
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
            )

        # Trim and decode
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
        raise RuntimeError(f"Error generating multi-image description: {e}")


def fine_tune(self, dataset_path: str, **kwargs) -> Dict[str, Any]:
    """Fine-tune Qwen2-VL model"""
    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import TrainingArguments, Trainer

        # Load dataset
        dataset = load_dataset("imagefolder", data_dir=dataset_path)

        # Prepare LoRA configuration
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        # Apply LoRA to model
        model_for_training = get_peft_model(self.model, lora_config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{Config.FINE_TUNED_MODELS_DIR}/qwen_finetuned",
            num_train_epochs=Config.EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            learning_rate=Config.LEARNING_RATE,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=100,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model_for_training,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", dataset["train"]),
            tokenizer=self.tokenizer,
        )

        # Start training
        trainer.train()

        # Save model
        trainer.save_model()

        return {
            "status": "success",
            "model_path": training_args.output_dir,
            "message": "Qwen2-VL fine-tuning completed successfully",
        }

    except Exception as e:
        logger.error(f"Error fine-tuning Qwen2-VL: {str(e)}")
        return {"status": "error", "message": f"Fine-tuning failed: {str(e)}"}
