from PIL import Image, ImageDraw
import face_recognition
import os


def recognize_faces_and_generate_prompt(
    image: Image.Image, known_faces_dir="data/faces"
) -> (Image.Image, str):
    """
    Recognize known people in the image and generate a descriptive prompt for VLM.
    """
    image_np = face_recognition.load_image_file(image.fp)

    # Detect and encode faces in the uploaded image
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    # Load known faces from disk
    known_encodings = []
    known_names = []
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            try:
                known_img = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(known_img)
                if encoding:
                    known_encodings.append(encoding[0])
                    known_names.append(person_name)
            except Exception:
                continue

    identified_names = []
    name_map = {}

    # Match faces
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "an unknown person"
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
        name_map[(top, right, bottom, left)] = name
        identified_names.append(name)

    # Annotate image (optional)
    draw = ImageDraw.Draw(image)
    for (top, right, bottom, left), name in name_map.items():
        draw.rectangle(((left, top), (right, bottom)), outline="green", width=2)
        draw.text((left, top - 10), name, fill="green")

    # Generate personalized prompt
    if identified_names:
        prompt = (
            "Describe what is happening in this image using the names of the individuals. "
            "Here are the people identified: " + ", ".join(set(identified_names)) + ". "
            "Use their names naturally in the description."
        )
    else:
        prompt = "Describe what is happening in this image, including people, objects, and actions."

    return image, prompt
