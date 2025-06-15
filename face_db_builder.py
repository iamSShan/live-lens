#!/usr/bin/env python3
"""
Automatic face database initialization script.
This directly processes all images from a folder structure and creates face_encodings.pkl file.
"""

import face_recognition
import pickle
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np


class DirectFaceDBBuilder:
    def __init__(self, output_path="face_encodings.pkl"):
        self.output_path = output_path
        self.known_face_encodings = []
        self.known_face_names = []

    def add_person_from_image(self, image_path, person_name):
        """Add a person directly from image file"""
        try:
            print(f"Processing {person_name} from {image_path}...")

            # Load image
            image = face_recognition.load_image_file(image_path)

            # Get face encodings
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                # Use the first face found
                encoding = face_encodings[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(person_name)
                print(f"Added {person_name}")

                if len(face_encodings) > 1:
                    print(f"Multiple faces found, using the first one")

                return True
            else:
                print(f"No face found in {image_path}")
                return False

        except Exception as e:
            print(f"Error processing {person_name}: {e}")
            return False

    def build_from_folder_structure(self, base_folder):
        """
        Build database from folder structure:
        base_folder/
        ├── Person1/
        │   ├── photo1.jpg
        │   └── photo2.jpg
        ├── Person2/
        │   └── photo.jpg
        """
        base_path = Path(base_folder)

        if not base_path.exists():
            print(f"Folder {base_folder} does not exist")
            return False

        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        total_processed = 0

        print(f"🔍 Scanning folder structure: {base_folder}")
        print("=" * 50)

        for person_folder in base_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                print(f"\nProcessing {person_name}...")

                # Find all images in this person's folder
                images = [
                    f
                    for f in person_folder.iterdir()
                    if f.suffix.lower() in supported_extensions
                ]

                if images:
                    # Process all images for this person (helps with accuracy)
                    person_images_processed = 0
                    for image_path in images:
                        if self.add_person_from_image(str(image_path), person_name):
                            person_images_processed += 1
                            total_processed += 1

                    print(
                        f"Processed {person_images_processed}/{len(images)} images for {person_name}"
                    )
                else:
                    print(f"No images found for {person_name}")

        print(f"\nTotal images processed: {total_processed}")
        return total_processed > 0

    def save_database(self):
        """Save the face database to file"""
        try:
            data = {
                "encodings": self.known_face_encodings,
                "names": self.known_face_names,
            }

            with open(self.output_path, "wb") as f:
                pickle.dump(data, f)

            print(f"\nDatabase saved to {self.output_path}")
            print(f"Total encodings in database: {len(self.known_face_encodings)}")

            # Show unique people count
            unique_people = list(set(self.known_face_names))
            print(f"Unique people: {len(unique_people)}")
            print(f"Names: {', '.join(unique_people)}")

            return True

        except Exception as e:
            print(f"Error saving database: {e}")
            return False


def auto_build_database(faces_folder="faces", output_file="face_encodings.pkl"):
    """
    Automatically build face database from folder structure

    Args:
        faces_folder (str): Path to folder containing person subfolders
        output_file (str): Output pickle file name
    """
    print("Automatic Face Recognition Database Builder")
    print("=" * 50)

    # Initialize builder
    builder = DirectFaceDBBuilder(output_file)

    # Check if faces folder exists
    if not os.path.exists(faces_folder):
        print(f"Faces folder '{faces_folder}' not found!")
        print(f"Please create the folder structure:")
        print(f"{faces_folder}/")
        print("├── Person1/")
        print("│   ├── photo1.jpg")
        print("│   └── photo2.jpg")
        print("├── Person2/")
        print("│   └── photo.jpg")
        print("└── ...")
        return False

    # Process all images
    success = builder.build_from_folder_structure(faces_folder)

    if success:
        # Save database
        if builder.save_database():
            print(f"\nFace database successfully created!")
            print(f"📁 File: {output_file}")
            return True
        else:
            print(f"\nFailed to save database")
            return False
    else:
        print(f"\nNo faces were processed")
        return False


if __name__ == "__main__":
    # Run it like: python face_db_builder.py
    # Default folder and output file
    faces_folder = "faces"
    output_file = "face_encodings.pkl"

    # Check command line arguments
    if len(sys.argv) >= 2:
        faces_folder = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    # Show usage
    print(f"Looking for faces in: {faces_folder}")
    print(f"Will save database to: {output_file}")
    print()

    # Run automatic processing
    success = auto_build_database(faces_folder, output_file)

    if success:
        print(f"\nDone! You can now use '{output_file}' for face recognition.")
    else:
        print(f"\nUsage: python {sys.argv[0]} [faces_folder] [output_file]")
        print(f"Example: python {sys.argv[0]} my_faces face_db.pkl")
