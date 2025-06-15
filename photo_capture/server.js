const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const app = express();
const port = 3000;

// Create photos directory if it doesn't exist
const photosDir = path.join(__dirname, '../faces');
if (!fs.existsSync(photosDir)) {
  fs.mkdirSync(photosDir);
}

// Serve static files from the 'public' folder
app.use(express.static(path.join(__dirname, 'public')));

// Set up multer for image upload
const upload = multer({ dest: 'uploads/' });

app.post('/save-photo', upload.single('photo'), (req, res) => {
  try {
    const username = req.body.username;
    const fullName = req.body.fullName || username;
    const folderPath = path.join(__dirname, '../faces', fullName);

    if (!fs.existsSync(folderPath)) {
      fs.mkdirSync(folderPath, { recursive: true });
    }

    const photoPath = path.join(folderPath, 'photo.jpg');
    fs.renameSync(req.file.path, photoPath);
    res.json({ success: true, message: 'Photo saved successfully' });
  } catch (error) {
    console.error('Error saving photo:', error);
    res.json({ success: false, message: 'Error saving photo' });
  }
});

// Add new route for folder check
app.get('/check-folder', (req, res) => {
  const username = req.query.username.toLowerCase();;
  const folderPath = path.join(__dirname, '../faces/', username);
  console.log(folderPath);

  if (fs.existsSync(folderPath)) {
    res.json({ exists: true });
  } else {
    res.json({ exists: false });
  }
});

// Start the server
app.listen(port, '127.0.0.1', () => { // Change '127.0.0.1' to '0.0.0.0' when running in server
  console.log(`Server running at http://0.0.0.0:${port}`);
});