# 📸 Live Lens

It is a real-time web application for webcam-based visual analysis powered by cutting-edge **Vision-Language Models (VLMs)** such as **SmolVLM**, **Qwen2-VL**, **GPT-4o**, **Gemini** and **PaLI-Gemma**. It can describe live scenes, identify known faces, and switch between models for comparison.


### 🖼️ Frontend Layout

[picture here]


## 🌟 Features

* Real-time webcam analysis with multiple vision models
* Model comparison - Switch between different vision models
* Auto-analysis mode for continuous monitoring
* Modern web interface with real-time updates
* Modular design for easy integration of new models  


## 🧠 Models used

### Local Models (Only Huggingface token required)
  * SmolVLM: Lightweight vision-language model from HuggingFace
  * Qwen2-VL: Advanced multimodal model from Alibaba
  * PaliGemma: Google’s image-to-text model (requires Hugging Face access).
  
    (These model are slow without GPU)

### API Models (Require API Keys)
  * GPT-4o: OpenAI's latest vision-capable model
  * Gemini: Google free model


## 🤳 Photo Capture
This is a simple tool that allows users to click a photo using their webcam and enter their name. The captured image and name are saved for use in a face recognition application, enabling the system to recognize and identify the user in future sessions.

How to run it:

Go inside photo_capture folder and run
```
  cd photo_capture
  npm install
  npm start
```

* Local URL: http://127.0.0.1:3000
* Or replace with your IP address and port: http://<your-ip>:3000


## ⚙️ Configuration

Edit config.py to modify:

* Update `CURRENT_MODEL` variable to use any other model
* Update `HOST` variable to "0.0.0.0" if running on any VM/server
* I have used Azure OpenAI, if you are using Azure too, then update ENDPOINT, API_VERSION, DEPLOYMENT, etc.
* Update `PORT` if required


## ⚡ Extra Note: 

> If you're using **OpenAI GPT-4** directly instead of **Azure GPT-4**, make sure to use `openai_without_azure.py` instead of `openai_model.py`. Set your OpenAI API key in .env file too.


## 🛠️ Setup
Follow the steps below to set up the application:
### 1. Create & Activate Virtual Environment

#### On **Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On **Mac/Linux**:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

If running on local without GPU:

```bash
pip install -r requirements_local.txt
```

If using GPU:

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory and add the necessary API keys:
- If using **GPT-4o**, **GPT-4**, or **OpenAI models**:
  ```env
  OPENAI_API_KEY=your_openai_key_here
  ```
- If using **Gemini**:
  ```env
  GEMINI_API_KEY=your_gemini_key_here
  ```
- If using **SmolVLM**, **Qwen**, or **PaliGemma**:
  ```env
  HUGGINGFACE_TOKEN=your_huggingface_token_here
  ```
      
### 4. Enable Face Recognition
If you want the app to identify the person in the webcam, follow these steps:

- Directly copy your photo into the `./live-lens/faces` folder. Create a folder with your name inside the `faces` folder and paste your image there.
- Alternatively, you can use the web app I’ve created for face registration.


**How to use the web app for face registration**:
1. Go to `photo_capture` directory and run `npm start`
2. Navigate to the web app (usually accessible at `http://localhost:3000` in local).
3. Follow the instructions to upload your photo and register your face.

For detailed instructions, please read this README file.

---

## 🚀 Run the application
```bash
python run.py
```

or we can directly run app.py:

In VM/Server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

In local:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and go to: http://localhost:8000 or http://<your-ip>:3000

## 📁 Project Structure

```bash
live-lens/
├── app.py                      # Main application entrypoint (FastAPI backend)
├── config.py                   # Configuration and environment setup
├── face_db_builder.py          # Processes images and saves face encodings
├── face_encodings.pkl          # Serialized file containing numerical face encodings
├── faces/                      # Saved face images
├── models/
│   ├── base_model.py           # Abstract base class for vision models
│   ├── face_recognition.py     # Recognizing faces in images using precomputed face encodings
│   ├── gemini_model.py         # Gemini LVM implementation
│   ├── openai_model.py         # GPT-4o LVM implementation
│   ├── paligemma.py            # Google paligemma-3b-mix-224 implementation
│   ├── qwen.py                 # Qwen2-VL-2B-Instruct implementation
│   ├── smolvlm.py              # SmolVLM-Instruct implementation
│   ├── utils.py                # Helper functions for the models
│   └──extra_func.py            # Some extra functions which are not getting used right now
├── photo_capture/              # Web app for face registration
├── frontend/
│   ├── index.html              # Main HTML page
│   ├── script.js               # JavaScript for frontend logic
│   └── style.css               # CSS styles
├── utils/
│   ├── image_utils.py          # Image utility script
│   ├── schemas.py              # Pydantic models for requests and responses 
├── requirements.txt            # Python dependencies list
├── .gitignore                  # Tells Git which files to ignore when commiting
├── .env                        # Environment variables, API keys, and config
└── run.py                      # Script to launch the application
```

## 🎛️ CUDA Support
We used CUDA 12.9, if torch is giving some issues with cuda/cpu, then install like this:

```bash
For CUDA support

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# For CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Dlib issues:
Sometimes while installing dlib libary which is used for face recognition may cause some issue like build failed or something, refer utils/dlib_issue.txt for the solution or reach out to me: https://in.linkedin.com/in/sharmashan


## 📈 Performance Tips

* **Use GPU (CUDA):**  
Models like **SmolVLM**, **Qwen**, and **PaliGemma** support GPU acceleration. If a GPU is available, it will significantly improve inference speed compared to CPU.

* **Initial Model Download:**  
The model is downloaded automatically on first use. This may take some time depending on the model size and your internet speed.

* **Frame Count Per Request:**  
Currently, the frontend (`frontend/script.js`) sends **5 frames** to the backend every time. You can experiment with increasing or decreasing this number based on your machine's performance and the quality of results.

* **Frame Send Interval:**  
Frames are sent from the frontend every **7 seconds** (`frontend/script.js`). Adjust this interval as needed depending on your use case, latency needs, and system capacity.


## 📚 References
- [Best Open Source Vision Model (Reddit Thread)](https://www.reddit.com/r/LocalLLaMA/comments/1dbge24/best_open_source_vision_model/)
- [SmolVLM: Efficient Vision-Language Models (Hugging Face Blog)](https://huggingface.co/blog/smolvlm)
- [PaliGemma: Vision-Language Model for Form and Table Understanding (Medium)](https://medium.com/@scholarly360/paligemma-vision-language-model-for-form-and-table-understanding-fff0cd48801b)
- [PaliGemma2-3B Model on Hugging Face](https://huggingface.co/google/paligemma2-3b-pt-224)
- [Qwen Vision-Language Model Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- [Gemini VLM](https://ai.google.dev/gemini-api/docs/image-understanding)
- [OpenAI Vision](https://platform.openai.com/docs/guides/images-vision?api-mode=responses)