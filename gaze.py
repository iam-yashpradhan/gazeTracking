import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize the mouse control variables
screen_width, screen_height = pyautogui.size()
mouse_x, mouse_y = screen_width // 2, screen_height // 2
mouse_speed = 60

# Function to move the mouse based on iris position
def move_mouse(left_eye_landmarks, right_eye_landmarks):
    global mouse_x, mouse_y
    left_iris_x = left_eye_landmarks[0].x
    left_iris_y = left_eye_landmarks[0].y
    right_iris_x = right_eye_landmarks[0].x
    right_iris_y = right_eye_landmarks[0].y
    mouse_x += int(((left_iris_x + right_iris_x) / 2 - 0.5) * mouse_speed)
    mouse_y += int(((left_iris_y + right_iris_y) / 2 - 0.5) * mouse_speed)
    mouse_x = max(0, min(mouse_x, screen_width))
    mouse_y = max(0, min(mouse_y, screen_height))
    pyautogui.moveTo(mouse_x, mouse_y)


cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract iris landmarks
            left_eye_indices = list(range(0, 160))
            right_eye_indices = list(range(160, 260))

            left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]

            # Move mouse based on iris position
            move_mouse(left_eye_landmarks, right_eye_landmarks)

            # Draw face mesh and iris landmarks
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()
