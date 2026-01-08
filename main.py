import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import cv2
import mediapipe as mp
import time # Import time for manual timestamping
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers

from dataclasses import dataclass, field
import numpy as np
from dataclasses import dataclass, field
import numpy as np
import time
from poses import WarriorIIPose, RaisedArmPose, PoseSmoother, CobraPose, SmallCobraPose
from voice_control import VoiceCommandCenter

# --- Drawing Constants ---
# Pairs of landmark indices that form the skeleton lines
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24),                   # Torso
    (23, 25), (25, 27), (24, 26), (26, 28)          # Legs
]

def draw_skeleton(image, landmarks, width, height, color=(0, 255, 0), thickness=2):
    """Draws the skeleton lines and joint points on the frame."""
    # 1. Draw the lines (Connections)
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        
        start_point = (int(start_lm.x * width), int(start_lm.y * height))
        end_point = (int(end_lm.x * width), int(end_lm.y * height))
        
        cv2.line(image, start_point, end_point, color, thickness)

    # 2. Draw the joints (Landmarks)
    for lm in landmarks:
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(image, (cx, cy), 4, (255, 255, 255), -1)

# --- 2. Pose Definitions ---

POSE_CLASSES = {
    "raised arm pose": RaisedArmPose,
    "warrior 2 pose": WarriorIIPose,
    "cobra pose": CobraPose,
    "small cobra pose": SmallCobraPose  # Added for fuzzy matching
}


# --- 3. Main Application ---
voice_center = VoiceCommandCenter(POSE_CLASSES)
# Start voice listening in a separate thread so video doesn't lag
voice_thread = threading.Thread(target=voice_center.listen_loop, daemon=True)
voice_thread.start()



# Initialize your dataclass-based coach
warrior_coach = RaisedArmPose()

# 1. Setup Configuration
model_path = 'pose_landmarker_lite.task' # Ensure this file is in your directory

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create the landmarker instance
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO # Optimized for frames
)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180 else angle



cap = cv2.VideoCapture(0)
start_time = time.time() # Record the start time
pose_smoother = PoseSmoother(alpha=0.6) # Tweak this for your Mac/Jetson latency


with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        # In your main loop, draw a 'Listening' indicator
        if not voice_center.stop_listening:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1) # Red dot like a recording light

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # ACCESS THE CLASS INSTANTIATED BY VOICE
        current_coach = voice_center.active_pose_instance
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            # landmarks = pose_smoother.smooth(raw_landmarks)

            is_perfect, coach_msg = current_coach.evaluate(landmarks)
            color = (0, 255, 0) if is_perfect else (0, 0, 255)
            thickness = 8 if is_perfect else 2

            draw_skeleton(frame, landmarks, w, h, color=color, thickness=thickness)
            # UI Feedbacks
            cv2.putText(frame, f"Active Pose: {current_coach.__class__.__name__}", (15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, coach_msg, (15, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Voice-Controlled Yoga Coach', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            voice_center.stop_listening = True
            break

cap.release()
cv2.destroyAllWindows()
