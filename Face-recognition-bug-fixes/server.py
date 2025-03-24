import cv2
import face_recognition
import os
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Store known face encodings and names
known_face_encodings = []
known_face_names = []

# Load images and get encodings
image_paths = [
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Maheshbabu.jpg", "Maheshbabu"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Messi.jpg", "Messi"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/abhinav.jpg", "Abhinav"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/ankit.jpg","Ankit"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/soumyajeet.jpg","Soumyajeet"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/nitish.jpg","nitish"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/prajwal.jpg","prajwal"),
]

for image_path, name in image_paths:
    if os.path.exists(image_path):
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)
            print(f"✅ Loaded: {name}")
        else:
            print(f"⚠️ No face found in {image_path}")
    else:
        print(f"❌ Error: File not found - {image_path}")

# Face recognition function
def recognize_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            return known_face_names[best_match_index]

    return "Unknown"

# Flask endpoint to receive frames and detect faces
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Decode the image from the request
    file = request.files['file']
    image_np = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Perform face recognition
    detected_name = recognize_faces(frame)

    return jsonify({"detected_name": detected_name})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
