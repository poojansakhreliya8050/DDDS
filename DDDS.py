import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
import winsound

# Constants
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20  # Number of consecutive frames the eye must be below the threshold
ALERT_SOUND_FILE = "alert.wav"  # Path to the alert sound file

# Load models
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(shape_predictor_path)

# Helper functions
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness(landmarks):
    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0
    return ear < EAR_THRESHOLD

# Initialize video capture
cap = cv2.VideoCapture(0)
frame_count = 0
is_alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    drowsy = False
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        try:
            if detect_drowsiness(landmarks):
                frame_count += 1
                if frame_count >= CONSEC_FRAMES and not drowsy and not is_alarm_playing:
                    print("drowsiness detected!")
                    winsound.PlaySound(ALERT_SOUND_FILE, winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)
                    is_alarm_playing = True
                    drowsy = True
            else:
                frame_count = 0
                if is_alarm_playing:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    is_alarm_playing = False
        except Exception as e:
            print(f"Error in drowsiness detection: {e}")

        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
