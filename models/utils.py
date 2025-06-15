import cv2
import numpy as np
from typing import Dict, List
from PIL import Image
from transformers import modeling_utils


# -------------------------------
# Fix for transformers 4.52.3
# -------------------------------
def apply_transformers_fix():
    """Apply monkey patch for transformers NoneType iteration bug"""
    original_post_init = modeling_utils.PreTrainedModel.post_init

    def patched_post_init(self):
        try:
            return original_post_init(self)
        except TypeError as e:
            if "argument of type 'NoneType' is not iterable" in str(e):
                return
            else:
                raise e

    modeling_utils.PreTrainedModel.post_init = patched_post_init


def preprocess_for_model(image_cv):
    # Convert BGR (OpenCV) to RGB (PIL)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def image_score(image):
    # Sharpness using Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness: Mean pixel intensity
    brightness = np.mean(gray)

    # Contrast: Standard deviation of pixel values
    contrast = np.std(gray)

    # Final weighted score (adjust weights as needed)
    score = 0.6 * sharpness + 0.2 * brightness + 0.2 * contrast
    return score


def select_best_frame(images):
    return max(images, key=image_score)


def create_single_image_person_context(faces: List[Dict], image: Image.Image) -> str:
    """Create context about people in a single image"""

    # Track unique people in the image
    unique_people = {}
    print(type(image))
    img_width, img_height = image.size

    # Process each face
    for face in faces:
        person_id = face.get("person_id", "unknown")
        name = face.get("name", "Unknown Person")
        confidence = face.get("confidence", 0)

        # Get face position information
        bbox = face.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox

        # Calculate relative position in the frame
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height

        # Determine position description
        position = ""
        if center_y < 0.33:
            position += "top "
        elif center_y > 0.66:
            position += "bottom "

        if center_x < 0.33:
            position += "left"
        elif center_x > 0.66:
            position += "right"
        else:
            position += "center"

        if position.strip() == "":
            position = "center"

        # Store person information
        if person_id not in unique_people:
            unique_people[person_id] = {
                "name": name,
                "position": position,
                "confidence": confidence,
            }

    # Generate context text
    context_parts = []

    # Overall summary
    person_count = len(unique_people)
    context_parts.append(
        f"There {'is' if person_count == 1 else 'are'} {person_count} {'person' if person_count == 1 else 'people'} in this image. "
    )

    # Person-specific information
    for person_id, info in unique_people.items():
        name = info["name"]
        position = info["position"]

        if "Unknown" in name:
            context_parts.append(f"A person is in the {position} of the image. ")
        else:
            context_parts.append(f"{name} is in the {position} of the image. ")

    return "".join(context_parts)


def create_multi_frame_person_context(
    all_faces: List[List[Dict]], images: List[Image.Image]
) -> str:
    """Create detailed context about multiple people across frames with positions"""

    # Track unique people across all frames
    unique_people = {}

    # Process each frame
    for frame_idx, faces in enumerate(all_faces):
        img_width, img_height = images[frame_idx].size

        for face in faces:
            person_id = face.get("person_id", "unknown")
            name = face.get("name", "Unknown Person")
            confidence = face.get("confidence", 0)

            # Get face position information
            bbox = face.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            # Calculate relative position in the frame
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height

            # Determine position description
            position = ""
            if center_y < 0.33:
                position += "top "
            elif center_y > 0.66:
                position += "bottom "

            if center_x < 0.33:
                position += "left"
            elif center_x > 0.66:
                position += "right"
            else:
                position += "center"

            if position.strip() == "":
                position = "center"

            # Store or update person information
            if person_id not in unique_people:
                unique_people[person_id] = {
                    "name": name,
                    "frames": [],
                    "confidence": confidence,
                }

            # Add frame appearance with position
            unique_people[person_id]["frames"].append(
                {"frame": frame_idx, "position": position, "bbox": bbox}
            )

    # Generate context text
    context_parts = []

    # Overall summary
    person_count = len(unique_people)
    context_parts.append(
        f"There {'is' if person_count == 1 else 'are'} {person_count} {'person' if person_count == 1 else 'people'} across these frames."
    )

    # Person-specific information
    for person_id, info in unique_people.items():
        name = info["name"]
        frames_present = [
            f["frame"] + 1 for f in info["frames"]
        ]  # 1-indexed for human readability

        if len(frames_present) == len(all_faces):
            presence = f"appears in all frames"
        else:
            presence = f"appears in frames {', '.join(map(str, frames_present))}"

        # Track movement across frames
        positions = [f["position"] for f in info["frames"]]
        if len(set(positions)) == 1:
            movement = f"stays in the {positions[0]} of the frame"
        else:
            movement = f"moves from {positions[0]} to {positions[-1]} across the frames"

        context_parts.append(f"{name} {presence} and {movement}.")

    return "\n".join(context_parts)
