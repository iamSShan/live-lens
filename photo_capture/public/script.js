// Accessing webcam
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const captureBtn = document.getElementById('capture-btn');
// const refreshBtn = document.getElementById('refresh-btn');
const retakeBtn = document.getElementById('retake-btn');
const saveBtn = document.getElementById('save-btn');
const photoPreview = document.getElementById('photo-preview');
const capturedPhoto = document.getElementById('captured-photo');
const usernameInput = document.getElementById('username');
const errorMessage = document.getElementById('error-message');

let currentStream = null;

// Capture the webcam feed
async function startWebcam() {
  try {
    console.log('Starting webcam...');
    currentStream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 },
        height: { ideal: 480 }
      } 
    });
    
    video.srcObject = currentStream;
    
    // Wait for video to load
    video.addEventListener('loadedmetadata', () => {
      console.log('Video metadata loaded');
      video.play().then(() => {
        console.log('Video playing');
        captureBtn.disabled = false;
        captureBtn.textContent = 'Capture Photo';
      }).catch(err => {
        console.error('Error playing video:', err);
      });
    });
    
  } catch (error) {
    console.error('Error accessing webcam:', error);
    errorMessage.textContent = 'Unable to access webcam. Please allow camera permissions.';
    captureBtn.textContent = 'Camera Error';
  }
}

// Stop webcam stream
function stopWebcam() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => {
      track.stop();
    });
    currentStream = null;
  }
}

// Capture a photo from the video feed
function capturePhoto() {
  console.log('Capturing photo...');
  console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
  
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    errorMessage.textContent = 'Webcam not ready. Please wait a moment.';
    return;
  }
  
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const photoData = canvas.toDataURL('image/jpeg', 0.8);
  capturedPhoto.src = photoData;

  // Stop webcam and hide video
  stopWebcam();
  video.style.display = 'none';
  
  // Show captured photo in the webcam container
  capturedPhoto.style.display = 'block';
  capturedPhoto.style.width = '100%';
  capturedPhoto.style.height = '100%';
  capturedPhoto.style.objectFit = 'cover';
  capturedPhoto.style.borderRadius = '10px';
  
  // Hide capture button, show retake/save buttons
  captureBtn.style.display = 'none';
  // refreshBtn.style.display = 'none'; // Hide Refresh button when retake/save buttons appear
  photoPreview.style.display = 'block';
  
  console.log('Photo captured successfully');
}

// Retake photo
retakeBtn.addEventListener('click', () => {
  console.log('Retaking photo...');
  
  // Hide captured photo and photo preview buttons
  capturedPhoto.style.display = 'none';
  photoPreview.style.display = 'none';
  
  // Show video and capture button
  video.style.display = 'block';
  captureBtn.style.display = 'block';
  captureBtn.disabled = true;
  captureBtn.textContent = 'Starting Camera...';
  
  startWebcam();
  // Show the Refresh button when Retake button is clicked
  // refreshBtn.style.display = 'inline-block';
});


// async function checkFolderExistence(username) {
//   // Mock API call to check folder existence on the server
//   const response = await fetch(`/check-folder?username=${username}`);
//   const data = await response.json();

//   if (data.exists) {
//     const fullName = prompt('This name already exists, please provide your full name:');
//     if (fullName) {
//       const fullNameResponse = await fetch(`/check-folder?username=${fullName}`);
//       const fullNameData = await fullNameResponse.json();

//       if (fullNameData.exists) {
//         const departmentOrShortForm = prompt('Full name already exists. Do you want to use short form or mention your department?');
//         return departmentOrShortForm;
//       } else {
//         return fullName;
//       }
//     }
//   }
//   return username;
// }

// Validate username before capturing photo
captureBtn.addEventListener('click', async (e) => {
  console.log('Capture button clicked');
  e.preventDefault();

  let username = usernameInput.value.trim();

  if (username === '') {
    errorMessage.textContent = 'Please enter your name!';
    return;
  }
  
  // Check folder existence and loop until a unique name is found
  username = await checkFolderExistence(username);
  // console.log(username);
  if (username === '') {
    errorMessage.textContent = 'No valid name provided.';
    return;
  }

  // After finding a valid, unique username, update the input field directly
  usernameInput.value = username;  // Update the username input field


  // Reset error message and capture button
  errorMessage.textContent = '';
  
  // Proceed to capture photo
  video.style.display = 'block';  // Show webcam
  capturedPhoto.style.display = 'none'; // Hide the captured photo preview
  photoPreview.style.display = 'none'; // Hide the retake/save buttons
  captureBtn.style.display = 'block'; // Show capture button
  captureBtn.disabled = false;
  captureBtn.textContent = 'Capture Photo';

  capturePhoto();
});


// Refresh webcam
// refreshBtn.addEventListener('click', () => {
//   console.log('Refreshing webcam...');
  
//   // Reset captured photo and buttons
//   capturedPhoto.style.display = 'none';
//   photoPreview.style.display = 'none';
//   video.style.display = 'block';
//   captureBtn.style.display = 'block';
//   captureBtn.disabled = true;
//   captureBtn.textContent = 'Starting Camera...';
  
//   // Restart webcam feed
//   startWebcam();
  
//   // Show the Refresh button when webcam feed is reloaded
//   refreshBtn.style.display = 'inline-block';
// });

// Save photo (handle backend interaction)
saveBtn.addEventListener('click', () => {
  const username = usernameInput.value.trim();
  const photoData = capturedPhoto.src;

  if (usernameExists(username)) {
    const fullName = prompt('Enter your full name to proceed:');
    if (fullName) {
      savePhotoToFolder(fullName, photoData);
    }
  } else {
    savePhotoToFolder(username, photoData);
  }
  
  // Reset the view after saving
  setTimeout(() => {
    // Hide captured photo and photo preview buttons
    capturedPhoto.style.display = 'none';
    photoPreview.style.display = 'none';
    
    // Show video and capture button
    video.style.display = 'block';
    captureBtn.style.display = 'block';
    captureBtn.disabled = true;
    captureBtn.textContent = 'Starting Camera...';
    
    // Clear the username input
    usernameInput.value = '';
    
    startWebcam();
  }, 1000); // Small delay to let user see the success message
});

// Mock function to check if the folder exists
function usernameExists(username) {
  return false;
}

// Function to save photo
function savePhotoToFolder(folderName, photoData) {
  // Convert base64 to blob
  const byteString = atob(photoData.split(',')[1]);
  const mimeString = photoData.split(',')[0].split(':')[1].split(';')[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  const blob = new Blob([ab], { type: mimeString });
  
  // Create FormData and send to server
  const formData = new FormData();
  formData.append('photo', blob, 'photo.jpg');
  formData.append('username', folderName);
  formData.append('fullName', folderName);
  
  fetch('/save-photo', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      // alert(`Photo saved successfully to folder: ${folderName}`);

    Swal.fire({
      title: "Photo saved successfully to folder",
      icon: "success",
      draggable: true
    });

    } else {
      alert('Error saving photo');
    }
  })
  .catch(error => {
    console.error('Error:', error);
    alert('Error saving photo');
  });
}

// Initialize webcam when page loads
document.addEventListener('DOMContentLoaded', () => {
  console.log('Page loaded, initializing...');
  captureBtn.disabled = true;
  captureBtn.textContent = 'Starting Camera...';
  startWebcam();
});

// New logic for handling folder existence and prompts:
// async function checkFolderExistence(username) {
//   // Mock API call to check folder existence on the server
//   const response = await fetch(`/check-folder?username=${username}`);
//   const data = await response.json();

//   if (data.exists) {
//     const fullName = prompt('Full name already exists, please provide your full name:');
//     if (fullName) {
//       const fullNameResponse = await fetch(`/check-folder?username=${fullName}`);
//       const fullNameData = await fullNameResponse.json();

//       if (fullNameData.exists) {
//         const departmentOrShortForm = prompt('Full name already exists. Do you want to use short form or mention your department?');
//         return departmentOrShortForm;
//       } else {
//         return fullName;
//       }
//     }
//   }
//   return username;
// }
async function checkFolderExistence(username) {
  let currentName = username;

  // Check if the folder for the current name exists
  let folderExists = await folderExistsOnServer(currentName);
  
  // Loop until we find a name that doesn't exist
  while (folderExists) {
    // Ask for the full name if the username exists
    currentName = prompt('This name already exists, please provide your full name:');

    if (!currentName) {
      return ''; // If user cancels or doesn't provide a full name, return empty
    }

    folderExists = await folderExistsOnServer(currentName);

    // If full name exists, ask for short form or department
    if (folderExists) {
      currentName = prompt('This full name already exists. Please provide a short form or mention your department:');
      
      if (!currentName) {
        return ''; // If user cancels or doesn't provide a short form/department, return empty
      }

      folderExists = await folderExistsOnServer(currentName);
    }
  }

  // Return the unique name once the folder doesn't exist
  return currentName;
}

// Helper function to check folder existence on the server
async function folderExistsOnServer(name) {
  try {
    const response = await fetch(`/check-folder?username=${name}`);
    const data = await response.json();
    return data.exists; // Return whether the folder exists
  } catch (error) {
    console.error('Error checking folder existence:', error);
    return false; // In case of error (e.g., server not responding), assume folder doesn't exist
  }
}
