import io
import base64
import logging
import cv2
import time
import numpy as np

# from fastapi.st
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from fastapi.staticfiles import StaticFiles

# Import models
# from models.smolvlm import SmolVLMModel
# from models.qwen import QwenVLModel
from models.openai_model import OpenAIVisionModel
from models.paligemma import PaliGemmaVisionModel
from models.smolvlm import SmolVLMModel
from models.qwen import QwenVLModel

# from models.gemini_model import GeminiVisionModel

# from .services.face_recognition_service import FaceRecognitionService

from utils.image_utils import base64_to_image, preprocess_for_model
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Live Lens", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


vision_model = None  # Global model reference


@app.on_event("startup")
async def startup_event():
    """
    Loading models when we start this application
    """
    global vision_model
    current_model = Config.CURRENT_MODEL.lower()

    try:
        if current_model == "gpt-4o":
            vision_model = OpenAIVisionModel()
        elif current_model == "smolvlm":
            vision_model = SmolVLMModel()
        elif current_model == "qwen":
            vision_model = QwenVLModel()
        elif current_model == "paligemma":
            vision_model = PaliGemmaVisionModel()
        else:
            raise ValueError(f"Unsupported model selected: {current_model}")

        vision_model.load_model()
        logging.info(f"{current_model} model loaded successfully at startup.")

    except Exception as e:
        logging.error(f"Failed to load model on startup: {str(e)}")
        raise


# Initialize models (lazy loading)
# models = {"smolvlm": None, "qwen": None, "gpt4o": None, "gemini": None}

# Initialize face recognition service
# face_service = FaceRecognitionService()


# Pydantic models
class ImageAnalysisRequest(BaseModel):
    image: str  # base64 encoded image
    model_name: str = "smolvlm"
    custom_prompt: Optional[str] = None
    include_face_recognition: bool = True


class ModelResponse(BaseModel):
    description: str
    faces: list = []
    model_used: str
    processing_time: float


def get_model(model_name: str):
    """Get or initialize the specified model"""
    global models

    if models[model_name] is None:
        logger.info(f"Initializing {model_name} model...")

        if model_name == "smolvlm":
            models[model_name] = SmolVLMModel()
        elif model_name == "qwen":
            models[model_name] = QwenVLModel()
        elif model_name == "gpt4o":
            models[model_name] = OpenAIVisionModel()
        elif model_name == "gemini":
            models[model_name] = GeminiVisionModel()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    return models[model_name]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    return FileResponse("frontend/index.html")


@app.post("/analyze", response_model=ModelResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze image with specified model"""

    start_time = time.time()

    try:
        # Convert base64 to image
        image_cv = base64_to_image(request.image)
        image_pil = preprocess_for_model(image_cv)

        # Get model
        # current_model = Config.CURRENT_MODEL
        # model = get_model(current_model)
        # model = "gpt4o"
        # Generate description

        description = vision_model.generate_description(
            image_pil, request.custom_prompt
        )

        # Face recognition if enabled
        # faces = []
        # if request.include_face_recognition:
        #     faces = face_service.recognize_faces(image_cv)

        processing_time = time.time() - start_time

        return ModelResponse(
            description=description,
            faces=[],
            model_used=request.model_name,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_available_models():
    """Get list of available models and their status"""
    model_status = {}
    for name, model_instance in models.items():
        model_status[name] = {
            "loaded": (
                model_instance is not None and model_instance.is_model_loaded()
                if model_instance
                else False
            ),
            "supports_fine_tuning": True,
        }
    return model_status


@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model"""
    try:
        model = get_model(model_name)
        if not model.is_model_loaded():
            model.load_model()
        return {"status": "success", "message": f"{model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Unload a specific model"""
    global models
    if model_name in models and models[model_name]:
        models[model_name].unload_model()
        models[model_name] = None
        return {"status": "success", "message": f"{model_name} unloaded successfully"}
    return {"status": "info", "message": f"{model_name} was not loaded"}


@app.post("/train/{model_name}")
async def train_model(model_name: str, dataset_path: str = "data/training"):
    """Fine-tune a model with custom dataset"""
    try:
        model = get_model(model_name)
        result = model.fine_tune(dataset_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
