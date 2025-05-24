// const video = document.getElementById('video');
// const summary = document.getElementById('summary');
// const faceCheckbox = document.getElementById('faceRecognition');

// let stream = null;
// let intervalId = null;

// // Start the camera and initiate auto-analysis every 5 seconds
// async function startCameraAndAutoAnalyze() {
//     try {
//         stream = await navigator.mediaDevices.getUserMedia({ video: true });
//         video.srcObject = stream;

//         intervalId = setInterval(() => {
//             analyzeFrame();
//         }, 5000);
//     } catch (err) {
//         console.error("Camera error:", err);
//         summary.textContent = "Failed to access webcam: " + err.message;
//     }
// }

// // Capture current frame and convert to base64
// function captureFrame() {
//     const canvas = document.createElement('canvas');
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     const context = canvas.getContext('2d');
//     context.drawImage(video, 0, 0);
//     return canvas.toDataURL('image/jpeg').split(',')[1]; // Remove data:image/jpeg;base64,
// }

// // Send image to FastAPI backend for analysis
// async function analyzeFrame() {
//     if (!video.videoWidth || !video.videoHeight) return;

//     const base64Image = captureFrame();
//     summary.textContent = 'Analyzing...';

//     try {
//         const response = await fetch('/analyze', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({
//                 image: base64Image,
//                 include_face_recognition: faceCheckbox.checked
//             })
//         });

//         const result = await response.json();

//         // if (response.ok) {
//         //     const faceInfo = result.faces?.length
//         //         ? result.faces.map(f => `• ${f.name} (x: ${f.location.left}, y: ${f.location.top})`).join('\n')
//         //         : "No faces detected";

//         //     summary.textContent =
//         //         `🧠 Model: ${result.model_used}\n⏱ Time: ${result.processing_time.toFixed(2)}s\n\n` +
//         //         result.description + '\n\n' +
//         //         (faceCheckbox.checked ? `😎 Faces:\n${faceInfo}` : '');
//         // } else {
//         //     summary.textContent = `Error: ${result.detail}`;
//         // }
//         if (response.ok) {
//             summary.textContent =
//                 `Model: ${result.model_used}\n⏱ Time: ${result.processing_time.toFixed(2)}s\n\n` +
//                 result.description;
//         } else {
//             summary.textContent = `Error: ${result.detail}`;
//         }
//     } catch (err) {
//         summary.textContent = `Request failed: ${err.message}`;
//     }
// }

// // Kick everything off when the page is ready
// window.addEventListener('DOMContentLoaded', () => {
//     startCameraAndAutoAnalyze();
// });

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
            analyzeFrame();
        }, 10000);
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

window.addEventListener('DOMContentLoaded', () => {
    startCameraAndAutoAnalyze();
});
