from pydantic import BaseModel
from typing import Optional, List


# Pydantic models
class ImageAnalysisRequest(BaseModel):
    """ To get request from frontend for single frame analysis """
    image: str  # base64 encoded image
    model_name: str = ""
    custom_prompt: Optional[str] = None
    include_face_recognition: bool = True


class MultiImageAnalysisRequest(BaseModel):
    """ To get request from frontend for multiple frames analysis """
    images: List[str]  # List of base64 encoded images
    model_name: str = ""
    custom_prompt: Optional[str] = None
    include_face_recognition: bool = True


class ModelResponse(BaseModel):
    """ For returning output back to frontend """
    description: str
    model_used: str
    processing_time: float
