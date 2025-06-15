# Photo Capture Web App

This is a web application that allows users to capture a photo from their webcam, validate their username to ensure uniqueness, and save the photo to a server-side folder. The app checks for username availability (case-insensitively) and allows users to retake or save their photos based on their unique username.

## Features

- **Webcam Access**: Captures photos directly from the user's webcam.
- **Username Validation**: Ensures that the username is unique by checking if a folder already exists with the same name.
- **Folder Management**: Organizes saved photos in user-specific folders on the server.
- **Photo Preview**: Allows users to preview the captured photo before saving or retaking.
- **Error Handling**: Displays helpful error messages when webcam access or folder creation fails.
- **Responsive UI**: The app is designed to be responsive and works on various screen sizes.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Backend**: Node.js with Express
- **File Management**: Multer for file upload
- **Photo Manipulation**: HTML Canvas
- **Alert System**: SweetAlert for user-friendly success/error messages

## Prerequisites

To run this project locally, you'll need to have the following software installed:

- [Node.js](https://nodejs.org/) (version >= 14)
- A modern web browser

## Setup Instructions

Follow the steps below to get the project up and running on your local machine.

### 1. Go to the directory

```bash
  cd photo_capture
```

### 2.  Install dependencies

```bash
  npm install
```

### 3. Start the server

```bash
  npm start
```

### 4. Access the app

- **Locally**: Open your browser and navigate to `http://localhost:3000`.

  The webcam will automatically start (ensure you have granted camera permissions in your browser).

- **On a Server**: If you're running the app on a server, you need to navigate to the public IP address or domain of the server (e.g., `http://your-server-ip:3000`).