const video = document.getElementById('video');
const summary = document.getElementById('summary');
const status = document.getElementById('status');  // 👈 Added status element
const faceCheckbox = document.getElementById('faceRecognition');

let stream = null;
let intervalId = null;

async function startCameraAndAutoAnalyze() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        intervalId = setInterval(() => {
            analyzeFrameMulti(); // Use `analyzeFrame` here to analyze just one frame each n secs
        }, 7000);
    } catch (err) {
        console.error("Camera error:", err);
        summary.textContent = "Failed to access webcam: " + err.message;
    }
}

function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg').split(',')[1];
}


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
        status.style.display = 'none';  // Hide "Analyzing..." only after everything is done
    }
}

async function analyzeFrameMulti() {
    if (!video.videoWidth || !video.videoHeight) return;

    const captureFrames = async (count = 5, interval = 400) => {
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
        const response = await fetch('/analyze-frames', { // Try with analyze-filtered too filter out best frame out of many
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


window.addEventListener('DOMContentLoaded', () => {
    startCameraAndAutoAnalyze();
});
