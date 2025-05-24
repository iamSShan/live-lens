# 📸 Live Lens

A comprehensive web application for real-time webcam analysis using multiple vision models including **SmolVLM**, **Qwen2-VL**, **GPT-4o**, and **PaliGemma**.

This is how it looks in frontend:
+----------------------+------------------------+
|                      |                        |
|    📷 Webcam Feed    |   🧠 Live Summary       |
|                      |                        |
+----------------------+------------------------+


## 🌟 Features

* Real-time webcam analysis with multiple vision models
* Model comparison - Switch between different vision models
* Auto-analysis mode for continuous monitoring
* Modern web interface with real-time updates
* Modular design for easy integration of new models  


## 🚀 Quick Start
```bash
# Clone or create the project directory
mkdir realtime-vision-app
cd realtime-vision-app

# Install Python dependencies
pip install -r requirements.txt

# Create .env file with your API keys
# Edit .env with your OpenAI and Huggingface API keys

# Run the Application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Access the App
Open your browser and go to: http://localhost:8000

## Models used

### Local Models (Only Huggingface token required)
  * SmolVLM: Lightweight vision-language model from HuggingFace
  * Qwen2-VL: Advanced multimodal model from Alibaba
  * PaliGemma: Google’s image-to-text model (requires Hugging Face access)

### API Models (Require API Keys)
  * GPT-4o: OpenAI's latest vision-capable model

## 🔧 Configuration
Edit config.py to modify:

Model parameters
* Image processing settings
* Fine-tuning configurations
* Server settings

## 🔑 Environment Variables
Create a .env file in the root directory:
```bash
# Required for GPT-4o
OPENAI_API_KEY=sk-your-openai-key-here

# Optional: For Hugging Face gated models
HUGGINGFACE_TOKEN=your-hf-token-here
```
<!-- # Optional: For private Hugging Face models
HUGGINGFACE_TOKEN=your-hf-token-here
📁 Project Structure
realtime-vision-app/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── models/              # Vision model implementations
│   │   ├── services/            # Business logic services
│   │   └── utils/               # Utility functions
├── data/
│   ├── faces/                   # Face recognition data
│   └── training/                # Fine-tuning datasets
├── models/                      # Saved model files
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── run.py                      # Startup script -->

## 🚨 Troubleshooting
CUDA/GPU Issues
```bash
For CUDA support

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```


## 📈 Performance Tips

GPU Usage: Ensure CUDA is properly installed for faster inference
Model Management: Unload unused models to save memory
Image Size: Resize large images before processing
Batch Processing: Process multiple images together when possible

## 🤝 Contributing

Fork the repository
Create a feature branch
Make your changes
Add tests if applicable
Submit a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support
If you encounter any issues:
* Check terminal logs for errors
* Ensure dependencies and API keys are set correctly

To report a bug, please include:

* Python version (recommended: 3.10)
* Your system details
* Exact error messages
* Steps to reproduce the issue


## 📚 References:
* https://www.reddit.com/r/LocalLLaMA/comments/1dbge24/best_open_source_vision_model/
* https://huggingface.co/blog/smolvlm
* https://medium.com/@scholarly360/paligemma-vision-language-model-for-form-and-table-understanding-fff0cd48801b
* https://huggingface.co/google/paligemma2-3b-pt-224
* https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e