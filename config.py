import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    VIDEO_SOURCE = "webcam"  # Other option is `video`

    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

    # Model names
    SMOLVLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"
    QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    PALIGEMMA_MODEL_NAME = "google/paligemma-3b-mix-224"
    GEMINI_MODEL_NAME = "gemini-1.5-flash-002"

    # OpenAI GPT-4o
    AZURE_OPENAI_ENDPOINT = ""  # Add your Azure endpoint if using Azure OpenAI
    AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
    AZURE_OPENAI_MODEL_NAME = "gpt-4o"
    AZURE_OPENAI_API_VERSION = ""  # Add your version if using Azure OpenAI

    # Current model in use
    CURRENT_MODEL = (
        "gemini"  # Other options are: `gpt-4o`, `paligemma`, `qwen` and `smolvlm`
    )

    # File paths
    FACE_DATA_DIR = "./faces"
    FACE_ENCODINGS_PATH = "./face_encodings.pkl"

    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
