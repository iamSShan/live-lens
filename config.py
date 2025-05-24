import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

    # Model configurations
    SMOLVLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"
    QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

    # File paths
    FACE_DATA_DIR = "data/faces"
    FACE_ENCODINGS_PATH = "models/face_encodings.pkl"
    FINE_TUNED_MODELS_DIR = "models/fine_tuned"

    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000

    # Vision settings
    MAX_IMAGE_SIZE = (800, 600)
    FACE_RECOGNITION_TOLERANCE = 0.6

    # Fine-tuning settings
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 4
    EPOCHS = 3
    MAX_LENGTH = 512

    # Model Specific Settings
    CURRENT_MODEL = "qwen"  # Other options are: gpt-4o

    # OpenAI GPT-4o
    AZURE_OPENAI_ENDPOINT = ""  # Add your Azure endpoint
    AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
    AZURE_OPENAI_MODEL_NAME = "gpt-4o"
    AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

    # GPT4_SETTINGS = {
    #     "max_tokens": int(os.environ.get("GPT4_MAX_TOKENS", "500")),
    #     "temperature": float(os.environ.get("GPT4_TEMPERATURE", "0.7")),
    #     "model": os.environ.get("GPT4_MODEL", "gpt-4o"),
    # }

    GEMINI_SETTINGS = {
        "max_tokens": int(os.environ.get("GEMINI_MAX_TOKENS", "500")),
        "temperature": float(os.environ.get("GEMINI_TEMPERATURE", "0.7")),
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp"),
    }

    SMOLVLM_SETTINGS = {
        "max_tokens": int(os.environ.get("SMOLVLM_MAX_TOKENS", "500")),
        "temperature": float(os.environ.get("SMOLVLM_TEMPERATURE", "0.7")),
        "model": os.environ.get("SMOLVLM_MODEL", "HuggingFaceTB/SmolVLM-Instruct"),
    }

    QWEN_SETTINGS = {
        "max_tokens": int(os.environ.get("QWEN_MAX_TOKENS", "500")),
        "temperature": float(os.environ.get("QWEN_TEMPERATURE", "0.7")),
        "model": os.environ.get("QWEN_MODEL", "Qwen/Qwen2-VL-7B-Instruct"),
    }
