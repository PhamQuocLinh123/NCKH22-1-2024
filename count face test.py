import numpy as np
import cv2

# Setting up the haar cascade classifiers from the OpenCV installation
face_cascade = cv2.CascadeClassifier("C:/Users/ADMIN/PycharmProjects/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/ADMIN/PycharmProjects/FlaskOpencv_FaceRecognition/resources/haarcascade_eye.xml")

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it to another index if needed)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Search for faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Initialize variables for tracking the face position
    face_x, face_y, face_w, face_h = 0, 0, 0, 0

    # For each face, detect eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Update the tracked face position
        face_x, face_y, face_w, face_h = x, y, w, h

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Update the face count
    face_count = len(faces)

    # Display the face count on the frame
    cv2.putText(frame, "So khuon mat: {}".format(face_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate the center of the detected face
    face_center_x = face_x + face_w // 2
    face_center_y = face_y + face_h // 2

    # Calculate the position for the camera frame to follow the face
    frame_height, frame_width, _ = frame.shape
    new_x = max(0, min(face_center_x - frame_width // 2, frame_width - frame_width))
    new_y = max(0, min(face_center_y - frame_height // 2, frame_height - frame_height))

    # Display the frame, adjusting the position to follow the face
    cv2.imshow('Face Detection', frame[new_y:new_y + frame_height, new_x:new_x + frame_width])

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
