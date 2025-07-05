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


def create_single_image_person_context(faces: List[Dict], image) -> str:
    """Create context about people in a single image"""

    # Track unique people in the image
    unique_people = {}

    # Get image dimensions based on image type
    if hasattr(image, "shape") and len(image.shape) >= 2:
        # NumPy array
        img_height, img_width = image.shape[:2]
    elif hasattr(image, "size") and isinstance(image.size, tuple):
        # PIL Image
        img_width, img_height = image.size
    else:
        # Default fallback
        img_width, img_height = 640, 480

    # Count occurrences of each name
    name_counts = {}
    for face in faces:
        name = face.get("name", "Unknown Person")
        name_counts[name] = name_counts.get(name, 0) + 1

    # Process each face
    for i, face in enumerate(faces):
        name = face.get("name", "Unknown Person")
        confidence = face.get("confidence", 0)

        # Get face position information
        location = face.get("location", {})
        if location:
            # If location is provided in the format with top, right, bottom, left
            x1 = location.get("left", 0)
            y1 = location.get("top", 0)
            x2 = location.get("right", 0)
            y2 = location.get("bottom", 0)
        else:
            # Try bbox format
            bbox = face.get("bbox", [0, 0, 0, 0])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0

        # Calculate relative position in the frame
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height

        # Determine position description
        if center_y < 0.33:
            if center_x < 0.33:
                position = "top left"
            elif center_x > 0.66:
                position = "top right"
            else:
                position = "top center"
        elif center_y > 0.66:
            if center_x < 0.33:
                position = "bottom left"
            elif center_x > 0.66:
                position = "bottom right"
            else:
                position = "bottom center"
        else:
            if center_x < 0.33:
                position = "middle left"
            elif center_x > 0.66:
                position = "middle right"
            else:
                position = "center"

        # Create a unique ID for this face based on position
        person_id = f"{name}_{position}_{i}"

        # Store person information
        unique_people[person_id] = {
            "name": name,
            "position": position,
            "confidence": confidence,
            "duplicate": name_counts[name]
            > 1,  # Flag if this name appears multiple times
        }

    # Generate context text
    context_parts = []

    # Overall summary
    person_count = len(unique_people)
    context_parts.append(
        f"There {'is' if person_count == 1 else 'are'} {person_count} {'person' if person_count == 1 else 'people'} in this image."
    )

    # Person-specific information
    for person_id, info in unique_people.items():
        name = info["name"]
        position = info["position"]
        is_duplicate = info["duplicate"]

        if "Unknown" in name:
            context_parts.append(f"A person is in the {position} of the image.")
        else:
            # If there are multiple people with the same name, include position
            if is_duplicate:
                context_parts.append(f"{name} is in the {position} of the image.")
            else:
                context_parts.append(f"{name} is in the {position} of the image.")

    return " ".join(context_parts)


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
            # Create a unique person_id if not present
            # Use a combination of name and location to differentiate people with the same name
            name = face.get("name", "Unknown Person")

            # Get face position information
            location = face.get("location", {})
            if location:
                # If location is provided in the format with top, right, bottom, left
                x1 = location.get("left", 0)
                y1 = location.get("top", 0)
                x2 = location.get("right", 0)
                y2 = location.get("bottom", 0)
            else:
                # Try bbox format
                bbox = face.get("bbox", [0, 0, 0, 0])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = 0, 0, 0, 0

            # Calculate center position for the first frame to create a unique ID
            if frame_idx == 0:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # Create a position-based ID to differentiate people with the same name
                position_id = f"{int(center_x/100)}_{int(center_y/100)}"
                person_id = f"{name}_{position_id}"
            else:
                # For subsequent frames, find the closest match from existing people
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                best_match = None
                min_distance = float("inf")

                for pid, info in unique_people.items():
                    if info["name"] == name and info["frames"]:
                        # Get the last known position
                        last_frame = info["frames"][-1]
                        last_bbox = last_frame.get("bbox", [0, 0, 0, 0])
                        last_x = (last_bbox[0] + last_bbox[2]) / 2
                        last_y = (last_bbox[1] + last_bbox[3]) / 2

                        # Calculate distance
                        distance = (
                            (center_x - last_x) ** 2 + (center_y - last_y) ** 2
                        ) ** 0.5

                        if distance < min_distance:
                            min_distance = distance
                            best_match = pid

                # If we found a close match, use that ID
                if (
                    best_match and min_distance < 200
                ):  # Threshold for considering it the same person
                    person_id = best_match
                else:
                    # Create a new person ID
                    position_id = f"{int(center_x/100)}_{int(center_y/100)}"
                    person_id = f"{name}_{position_id}_{frame_idx}"

            confidence = face.get("confidence", 0)

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
                {
                    "frame": frame_idx,
                    "position": position,
                    "bbox": [x1, y1, x2, y2],  # Store bbox for tracking
                }
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

        # For multiple people with the same name, add position qualifier
        if sum(1 for pid, pinfo in unique_people.items() if pinfo["name"] == name) > 1:
            # Get the most common position for this person
            positions_count = {}
            for f in info["frames"]:
                pos = f["position"]
                positions_count[pos] = positions_count.get(pos, 0) + 1

            most_common_position = max(positions_count.items(), key=lambda x: x[1])[0]
            name = f"{name} in the {most_common_position}"

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
