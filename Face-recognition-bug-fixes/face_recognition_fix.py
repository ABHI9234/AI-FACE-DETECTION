# # main code
# import cv2
# import face_recognition
#
# # Store face encodings and names
# known_face_encodings = []
# known_face_names = []
#
# # Load images and get encodings
# image_paths = [
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Maheshbabu.jpg", "Maheshbabu"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Messi.jpg", "Messi"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/abhinav.jpg","Abhinav"),
# ]
# for image_path, name in image_paths:
#     image = face_recognition.load_image_file(image_path)
#     encoding = face_recognition.face_encodings(image)
#     if encoding:  # Ensure an encoding is found
#         known_face_encodings.append(encoding[0])
#         known_face_names.append(name)
#
# # Open webcam
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = video_capture.read()
#
#     # Detect face locations and encodings in the frame
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Compare with known faces
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         name = "Unknown"
#
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
#
#         # Draw a box and label on the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     # Display the video frame
#     cv2.imshow("Video", frame)
#
#     # Exit loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Clean up
# video_capture.release()
# cv2.destroyAllWindows()


# # for knife
# import cv2
# import face_recognition
# from ultralytics import YOLO
#
# # Load YOLO model for weapon detection
# model = YOLO("yolov8n.pt")
#
# # Store face encodings and names
# known_face_encodings = []
# known_face_names = []
#
# # Load images and get encodings
# image_paths = [
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Maheshbabu.jpg", "Maheshbabu"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Messi.jpg", "Messi"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/abhinav.jpg", "Abhinav"),
# ]
#
# for image_path, name in image_paths:
#     image = face_recognition.load_image_file(image_path)
#     encoding = face_recognition.face_encodings(image)
#     if encoding:  # Ensure an encoding is found
#         known_face_encodings.append(encoding[0])
#         known_face_names.append(name)
#
# # Open webcam
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = video_capture.read()
#
#     # Detect face locations and encodings in the frame
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#     # Perform weapon detection using YOLO
#     results = model(frame)
#
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Compare with known faces
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         name = "Unknown"
#
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
#
#         # Draw a box and label on the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     # Draw weapon detection results
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             label = model.names[int(box.cls)]
#
#             if label in ["knife", "gun"]:
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     # Display the video frame
#     cv2.imshow("Video", frame)
#
#     # Exit loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Clean up
# video_capture.release()
# cv2.destroyAllWindows()

# with voice many times
# import cv2
# import face_recognition
# import os
#
# # Speak function using macOS's 'say' command
# def speak(text):
#     os.system(f"say '{text}'")
#
# # Store face encodings and names
# known_face_encodings = []
# known_face_names = []
#
# # Load images and get encodings
# image_paths = [
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Maheshbabu.jpg", "Maheshbabu"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Messi.jpg", "Messi"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/abhinav.jpg","Abhinav"),
# ]
# for image_path, name in image_paths:
#     image = face_recognition.load_image_file(image_path)
#     encoding = face_recognition.face_encodings(image)
#     if encoding:
#         known_face_encodings.append(encoding[0])
#         known_face_names.append(name)
#
# # Open webcam
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = video_capture.read()
#
#     # Detect face locations and encodings
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         name = "Unknown"
#
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
#
#         # Estimate distance based on face size (smaller face = further away)
#         face_height = bottom - top
#         if face_height > 250:
#             distance = "very close"
#         elif face_height > 150:
#             distance = "a few meters away"
#         else:
#             distance = "far away"
#
#         # Speak the name and distance
#         speak(f"{name} is {distance}.")
#
#         # Draw box and label
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.putText(frame, f"{name}, {distance}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     # Display video frame
#     cv2.imshow("Video", frame)
#
#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()

# with voice twice
import cv2
import face_recognition
import os

# Speak function using macOS's 'say' command
def speak(text):
    os.system(f"say '{text}'")

# Store face encodings and names
known_face_encodings = []
known_face_names = []

# Track how many times each name is spoken
speak_count = {}

# Load images and get encodings
image_paths = [
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Maheshbabu.jpg", "Maheshbabu"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Messi.jpg", "Messi"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/abhinav.jpg","Abhinav"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/ankit.jpg","Ankit"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/soumyajeet.jpg","Soumyajeet"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/nitish.jpg","nitish"),
    ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/prajwal.jpg","prajwal"),
]
for image_path, name in image_paths:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(name)
        speak_count[name] = 0

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Estimate distance based on face size (smaller face = further away)
        face_height = bottom - top
        if face_height > 250:
            distance = "very close"
        elif face_height > 150:
            distance = "a few meters away"
        else:
            distance = "far away"

        # Speak the name and distance (only twice per person)
        if name != "Unknown" and speak_count[name] < 2:
            speak(f"{name} is {distance}.")
            speak_count[name] += 1

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, f"{name}, {distance}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display video frame
    cv2.imshow("Video", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# import cv2
# import face_recognition
# import os
# import json
#
# # Speak function using macOS's 'say' command
# def speak(text):
#     os.system(f"say '{text}'")
#
# # Store face encodings and names
# known_face_encodings = []
# known_face_names = []
#
# # Track how many times each name is spoken
# speak_count = {}
#
# # Store recognized faces for JSON output
# recognized_faces = []
#
# # Load images and get encodings
# image_paths = [
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Maheshbabu.jpg", "Maheshbabu"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Messi.jpg", "Messi"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/abhinav.jpg", "Abhinav"),
# ]
# for image_path, name in image_paths:
#     image = face_recognition.load_image_file(image_path)
#     encoding = face_recognition.face_encodings(image)
#     if encoding:
#         known_face_encodings.append(encoding[0])
#         known_face_names.append(name)
#         speak_count[name] = 0
#
# # Open webcam
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = video_capture.read()
#
#     # Detect face locations and encodings
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         name = "Unknown"
#
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
#
#         # Estimate distance based on face size (smaller face = further away)
#         face_height = bottom - top
#         if face_height > 250:
#             distance = "very close"
#         elif face_height > 150:
#             distance = "a few meters away"
#         else:
#             distance = "far away"
#
#         # Speak the name and distance (only twice per person)
#         if name != "Unknown" and speak_count[name] < 2:
#             speak(f"{name} is {distance}.")
#             speak_count[name] += 1
#
#         # Draw box and label
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.putText(frame, f"{name}, {distance}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#         # Store recognized face information for JSON output
#         recognized_faces.append({
#             "name": name,
#             "distance": distance,
#             "location": {"top": top, "right": right, "bottom": bottom, "left": left}
#         })
#
#     # Display video frame
#     cv2.imshow("Video", frame)
#
#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Save recognized faces to a JSON file
# with open("recognized_faces.json", "w") as json_file:
#     json.dump(recognized_faces, json_file, indent=4)
#
# print("✅ Recognized face data saved to 'recognized_faces.json'")
#
# video_capture.release()
# cv2.destroyAllWindows()
#

# import cv2
# import face_recognition
# import os
#
# # Speak function using macOS's 'say' command
# def speak(text):
#     os.system(f"say '{text}'")
#
# # Store face encodings and names
# known_face_encodings = []
# known_face_names = []
#
# # Track how many times each name is spoken
# speak_count = {}
#
# # Load images and get encodings
# image_paths = [
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Maheshbabu.jpg", "Maheshbabu"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/Messi.jpg", "Messi"),
#     ("/Users/abhinavtadiparthi/Desktop/Face-recognition-bug-fixes/Images/abhinav.jpg", "Abhinav"),
# ]
# for image_path, name in image_paths:
#     image = face_recognition.load_image_file(image_path)
#     encoding = face_recognition.face_encodings(image)
#     if encoding:
#         known_face_encodings.append(encoding[0])
#         known_face_names.append(name)
#         speak_count[name] = 0
#
# # Open webcam
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = video_capture.read()
#
#     # Detect face locations and encodings
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         name = "Unknown"
#
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
#
#         # Estimate distance based on face size (smaller face = further away)
#         face_height = bottom - top
#         if face_height > 250:
#             distance = "very close"
#         elif face_height > 150:
#             distance = "a few meters away"
#         else:
#             distance = "far away"
#
#         # Speak the name and distance (only twice per person)
#         if name != "Unknown" and speak_count[name] < 2:
#             speak(f"{name} is {distance}.")
#             speak_count[name] += 1
#
#         # Draw box and label on the video feed
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.putText(frame, f"{name}, {distance}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#         # Print recognized face details in terminal
#         print(f"Detected: {name} - {distance} (Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left})")
#
#     # Display video frame
#     cv2.imshow("Video", frame)
#
#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()
#
#


