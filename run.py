#!/usr/bin/env python3
import os
import sys
import subprocess
import importlib.util
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("LiveLens-Runner")


def import_config():
    """Import the config module dynamically"""
    try:
        spec = importlib.util.spec_from_file_location("config", "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config.Config  # Return the Config class
    except Exception as e:
        logger.error(f"Failed to import config.py: {e}")
        sys.exit(1)


def check_face_encodings():
    """Check if face encodings file exists"""
    config = import_config()
    face_db_path = getattr(config, "FACE_ENCODINGS_PATH", "./face_encodings.pkl")

    if not os.path.exists(face_db_path):
        logger.info(f"Face encodings file not found at {face_db_path}")
        return False

    logger.info(f"Face encodings file found at {face_db_path}")
    return True


def build_face_database():
    """Run the face database builder script"""
    logger.info("Building face database...")

    try:
        face_db_builder_path = "face_db_builder.py"
        if not os.path.exists(face_db_builder_path):
            logger.error(
                f"Face database builder script not found at {face_db_builder_path}"
            )
            sys.exit(1)

        result = subprocess.run(
            [sys.executable, face_db_builder_path],
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info("Face database built successfully")
        logger.debug(result.stdout)

        if result.stderr:
            logger.warning(f"Warnings during face database build: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build face database: {e}")
        logger.error(f"Error output: {e.stderr}")
        sys.exit(1)


def check_env_requirements():
    """Check if .env file exists and has required API keys based on the model"""
    if not os.path.exists(".env"):
        logger.error(
            ".env file not found. Please create a .env file with required API keys."
        )
        sys.exit(1)

    # Load environment variables from .env
    load_dotenv()

    # Get current model from config
    config = import_config()
    current_model = getattr(config, "CURRENT_MODEL", "").lower()

    logger.info(f"Detected model configuration: {current_model}")

    # Check for required API keys based on model
    if current_model in ["smolvlm", "qwen", "paligemma"]:
        if not os.environ.get("HUGGINGFACE_TOKEN"):
            logger.error(
                f"HUGGINGFACE_TOKEN is required for {current_model} but not found in .env file"
            )
            sys.exit(1)
        else:
            logger.info(f"HUGGINGFACE_TOKEN found for {current_model}")

    elif "gemini" in current_model:
        if not os.environ.get("GEMINI_API_KEY"):
            logger.error(
                "GEMINI_API_KEY is required for Gemini models but not found in .env file"
            )
            sys.exit(1)
        else:
            logger.info("GEMINI_API_KEY found")

    elif "gpt" in current_model or "openai" in current_model:
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error(
                "OPENAI_API_KEY is required for OpenAI models but not found in .env file"
            )
            sys.exit(1)
        else:
            logger.info("OPENAI_API_KEY found")

    logger.info("Environment configuration validated successfully")


def start_fastapi_app():
    """Start the FastAPI application using uvicorn"""
    config = import_config()
    current_model = getattr(config, "CURRENT_MODEL", "Unknown")
    host = getattr(config, "HOST", "0.0.0.0")
    port = getattr(config, "PORT", 8000)

    logger.info(f"Starting LiveLens with model: {current_model}")
    logger.info(f"Server will be available at http://{host}:{port}")

    try:
        # Use uvicorn programmatically
        import uvicorn

        uvicorn.run("app:app", host=host, port=port, reload=True, log_level="info")
    except ImportError:
        # Fall back to subprocess if uvicorn can't be imported
        logger.info("Starting server using subprocess...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            check=True,
        )


def main():
    """Main function to run the application"""
    logger.info("Starting Live Lens initialization...")

    # Check environment variables
    check_env_requirements()

    # Check if face encodings exist, build if not
    if not check_face_encodings():
        build_face_database()

    # Start the FastAPI application
    start_fastapi_app()


if __name__ == "__main__":
    main()
