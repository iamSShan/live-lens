const video = document.getElementById('video');
const summary = document.getElementById('summary');
const status = document.getElementById('status');
const faceCheckbox = document.getElementById('faceRecognition');
const videoFileInput = document.getElementById('videoFile');

let intervalId = null;

// When the user selects a video, load it into the video player
videoFileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const videoURL = URL.createObjectURL(file);
        video.src = videoURL;
        video.play();
        summary.textContent = "Video loaded. Ready to analyze!";
        
        // When the video starts playing, begin analysis
        video.onplay = () => {
            summary.textContent = "Video is playing. Analyzing frames...";
            startVideoAndAutoAnalyze(); // Start the analysis process
        };
    }
});

// Function to capture and analyze a single frame
function captureFrame(scaledWidth = 480) {
    const aspectRatio = video.videoWidth / video.videoHeight;
    const width = scaledWidth;
    const height = Math.round(width / aspectRatio);

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, width, height);

    return canvas.toDataURL('image/jpeg', 0.6).split(',')[1]; // 0.6 = reduce quality slightly for speed
}

// Analyze a single frame (for face recognition, etc.)
async function analyzeFrame() {
    if (!video.videoWidth || !video.videoHeight) return;

    const base64Image = captureFrame();
    status.style.display = 'block';  // Show "Analyzing..."

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64Image,
                include_face_recognition: faceCheckbox.checked
            })
        });

        const result = await response.json();

        if (response.ok) {
            summary.textContent =
                `🧠 Model: ${result.model_used}\n⏱ Time: ${result.processing_time.toFixed(2)}s\n\n` +
                result.description;
        } else {
            summary.textContent = `Error: ${result.detail}`;
        }
    } catch (err) {
        summary.textContent = `Request failed: ${err.message}`;
    } finally {
        status.style.display = 'none';  // Hide "Analyzing..."
    }
}

// Analyze multiple frames (if required)
async function analyzeFrameMulti() {
    if (!video.videoWidth || !video.videoHeight) return;

    const captureFrames = async (count = 5, interval = 700) => {
        const frames = [];
        for (let i = 0; i < count; i++) {
            frames.push(captureFrame());
            await new Promise(res => setTimeout(res, interval)); // wait before next capture
        }
        return frames;
    };

    const base64Frames = await captureFrames();

    status.style.display = 'block'; // Show analyzing

    try {
        const response = await fetch('/analyze-frames', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                images: base64Frames,
                include_face_recognition: faceCheckbox.checked
            })
        });

        const result = await response.json();

        if (response.ok) {
            summary.textContent =
                `📸 Multi-Frame\n🧠 Model: ${result.model_used}\n⏱ Time: ${result.processing_time.toFixed(2)}s\n\n` +
                result.description;
        } else {
            summary.textContent = `Error: ${result.detail}`;
        }
    } catch (err) {
        summary.textContent = `Request failed: ${err.message}`;
    } finally {
        status.style.display = 'none';
    }
}

// Start analyzing video frames periodically
function startVideoAndAutoAnalyze() {
    intervalId = setInterval(() => {
        analyzeFrameMulti(); // Analyze a batch of frames every 2 seconds
    }, 5000);
}
