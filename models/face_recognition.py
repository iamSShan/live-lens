from PIL import Image, ImageDraw
import face_recognition
import os
import numpy as np
from typing import Dict, Any, List
import logging
import pickle

logger = logging.getLogger(__name__)


class FaceRecognitionManager:
    """Handles face recognition and encoding management"""

    def __init__(self, encodings_path: str = "face_encodings.pkl"):
        self.encodings_path = encodings_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()

    def load_encodings(self):
        """Load pre-computed face encodings from file"""
        try:
            if os.path.exists(self.encodings_path):
                with open(self.encodings_path, "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
                logger.info(f"Loaded {len(self.known_face_names)} face encodings")
            else:
                logger.warning(f"No encodings file found at {self.encodings_path}")
        except Exception as e:
            logger.error(f"Error loading face encodings: {e}")

    def save_encodings(self):
        """Save face encodings to file"""
        try:
            data = {
                "encodings": self.known_face_encodings,
                "names": self.known_face_names,
            }
            with open(self.encodings_path, "wb") as f:
                pickle.dump(data, f)
            logger.info("Face encodings saved successfully")
        except Exception as e:
            logger.error(f"Error saving face encodings: {e}")

    def add_person(self, image_path: str, person_name: str):
        """Add a new person to the recognition database"""
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(person_name)
                self.save_encodings()
                logger.info(f"Added {person_name} to face recognition database")
                return True
            else:
                logger.warning(f"No face found in image for {person_name}")
                return False
        except Exception as e:
            logger.error(f"Error adding person {person_name}: {e}")
            return False

    def recognize_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Recognize faces in the given image"""
        try:
            # Convert PIL Image to numpy array
            image_array = np.array(image)

            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image_array)
            face_encodings = face_recognition.face_encodings(
                image_array, face_locations
            )

            recognized_faces = []

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=0.6
                )
                name = "Unknown"
                confidence = 0.0

                # Find the best match
                if True in matches:
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]

                recognized_faces.append(
                    {
                        "name": name,
                        "confidence": confidence,
                        "location": {
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "left": left,
                        },
                    }
                )

            return recognized_faces

        except Exception as e:
            logger.error(f"Error recognizing faces: {e}")
            return []
