import io
import base64
import logging
import time
from PIL import Image

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Import models
from models.smolvlm import SmolVLMModel
from models.qwen import QwenVLModel
from models.paligemma import PaliGemmaVisionModel
from models.openai_model import OpenAIVisionModel
from models.gemini_model import GeminiVisionModel

# Import helper functions and config file
from utils.image_utils import base64_to_image, preprocess_for_model
from utils.schemas import ImageAnalysisRequest, MultiImageAnalysisRequest, ModelResponse
from config import Config


# Configure logging for better traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Live Lens", version="1.0.0")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

video_source = Config.VIDEO_SOURCE.lower()  # "webcam" or "video"


# Mount the frontend directory to serve static files
if video_source == "webcam":
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
else:
    app.mount(
        "/frontend_vid", StaticFiles(directory="frontend_vid"), name="frontend_vid"
    )

# Global reference to the current vision model
vision_model = None


@app.on_event("startup")
async def startup_event():
    """
    Load the selected model at application startup.
    Based on the config, it selects the appropriate model and loads it.
    """
    global vision_model
    current_model = Config.CURRENT_MODEL.lower()

    try:
        # Load the selected model based on the config value
        if current_model == "gpt-4o":
            vision_model = OpenAIVisionModel()
        elif current_model == "smolvlm":
            vision_model = SmolVLMModel()
        elif current_model == "qwen":
            vision_model = QwenVLModel()
        elif current_model == "gemini":
            vision_model = GeminiVisionModel()
        elif current_model == "paligemma":
            vision_model = PaliGemmaVisionModel()
        else:
            raise ValueError(f"Unsupported model selected: {current_model}")

        # Call load_model to initialize the model
        vision_model.load_model()
        logging.info(f"{current_model} model loaded successfully at startup.")

    except Exception as e:
        # Log and raise error if model loading fails
        logging.error(f"Failed to load model on startup: {str(e)}")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint that serves the main HTML page for the application.
    This determines which frontend UI to serve based on the VIDEO_SOURCE setting

    - If VIDEO_SOURCE is set to "webcam", it serves the webcam-based frontend (frontend/index.html).
    - If VIDEO_SOURCE is set to "video", it serves the video-based frontend (frontend_vid/index.html).
    - If VIDEO_SOURCE is invalid or not recognized, it returns a 400 Bad Request.
    """
    if video_source == "webcam":
        # If we want to summarize through webcam
        return FileResponse("frontend/index.html")
    elif video_source == "video":
        # If we want to summarize video after uploading
        return FileResponse("frontend_vid/index.html")
    else:
        raise HTTPException(
            status_code=400, detail="Invalid VIDEO_SOURCE configuration."
        )


@app.post("/analyze", response_model=ModelResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze a single image and generate a description using the selected model.
    Measures processing time for performance tracking.
    This endpoint is not getting used right now
    """

    start_time = time.time()
    try:
        # Convert base64 to image
        image_cv = base64_to_image(request.image)
        image_pil = preprocess_for_model(image_cv)

        description = vision_model.generate_description(
            image_pil, request.custom_prompt
        )

        processing_time = time.time() - start_time

        return ModelResponse(
            description=description,
            model_used=request.model_name,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-frames", response_model=ModelResponse)
async def analyze_frames(request: MultiImageAnalysisRequest):
    """
    Analyze multiple images (frames) and generate descriptions using the selected model.
    Measures processing time for performance tracking.
    """
    start = time.time()
    try:
        # Convert base64-encoded images to image objects (cv2 format)
        images = [base64_to_image(img) for img in request.images]
        logger.info(f"Face recognition: {request.include_face_recognition}")

        # Based on some observation, these variants of Qwen, SmolVLM doesn't works well when multiple images are passed together
        # So we will pass the best image(frame) to them
        # And for OpenAI GPT4o and Gemini, we will be passing multiple images as it works fine

        # Generate a description based on the images and whether face recognition is enabled
        description = vision_model.generate_description(
            images, request.include_face_recognition
        )

        duration = time.time() - start
        logger.info(f"Generated response in: {duration} seconds")

        # Return the generated description and processing time in the response
        return ModelResponse(
            description=description,
            model_used=Config.CURRENT_MODEL,
            processing_time=duration,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
