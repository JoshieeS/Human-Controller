# controller.py

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Adjust these to tune sensitivity, smoothing, and gesture timing
SENSITIVITY = 5
SMOOTHING_FACTOR = 0.9
DEADZONE_RATIO = 0.01  # 1% of screen width/height
HISTORY_LENGTH = 5     # number of frames to average gaze over

HOLD_THRESHOLD = 0.6   # hold pinch >0.6s → right-click
DOUBLE_WINDOW = 0.4    # two quick pinches <0.4s apart → double-click

# ─── 1) Initialize MediaPipe modules ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,       # includes iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils  # for drawing connections & landmarks

# ─── 2) Get screen size & initialize cursor ─────────────────────────────────
screen_w, screen_h = pyautogui.size()
pyautogui.moveTo(screen_w // 2, screen_h // 2, duration=0)

# ─── 3) Helper: distance between two landmarks ────────────────────────────────
def landmark_distance(lm1, lm2, img_w, img_h):
    x1, y1 = int(lm1.x * img_w), int(lm1.y * img_h)
    x2, y2 = int(lm2.x * img_w), int(lm2.y * img_h)
    return np.hypot(x2 - x1, y2 - y1)

# ─── 4) Setup smoothing history ───────────────────────────────────────────────
gaze_history = deque(maxlen=HISTORY_LENGTH)
last_cursor_x, last_cursor_y = screen_w // 2, screen_h // 2

# ─── 5) Pinch state & timing variables ───────────────────────────────────────
pinch_active = False
pinch_start_time = 0
last_quick_pinch_time = 0

# ─── 6) Open webcam ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # uncomment to fix height

prev_scroll_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror & convert for MediaPipe
    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== 6.1) FaceMesh: detect face landmarks & draw connections ==============
    face_results = face_mesh.process(rgb)
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]

        # Draw all face‐mesh connections (green/blue)
        mp_draw.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
        )

        # Refined iris indices
        LEFT_IRIS  = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        # Compute each iris center in normalized coords
        lx = np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS])
        ly = np.mean([face_landmarks.landmark[i].y for i in LEFT_IRIS])
        rx = np.mean([face_landmarks.landmark[i].x for i in RIGHT_IRIS])
        ry = np.mean([face_landmarks.landmark[i].y for i in RIGHT_IRIS])
        gaze_x = (lx + rx) / 2
        gaze_y = (ly + ry) / 2

        # Append to history + average for smoothing
        gaze_history.append((gaze_x, gaze_y))
        avg_gx = sum(v[0] for v in gaze_history) / len(gaze_history)
        avg_gy = sum(v[1] for v in gaze_history) / len(gaze_history)

        # Apply sensitivity multiplier on offset from center
        norm_dx = (avg_gx - 0.5) * SENSITIVITY
        norm_dy = (avg_gy - 0.5) * SENSITIVITY
        nx = max(0.0, min(1.0, 0.5 + norm_dx))
        ny = max(0.0, min(1.0, 0.5 + norm_dy))

        # Map to screen coords
        target_x = int(nx * screen_w)
        target_y = int(ny * screen_h)

        # Deadzone: only move if beyond threshold
        dx = abs(target_x - last_cursor_x)
        dy = abs(target_y - last_cursor_y)
        if dx > DEADZONE_RATIO * screen_w or dy > DEADZONE_RATIO * screen_h:
            # Smoothly ease toward target
            smooth_x = int(last_cursor_x + (target_x - last_cursor_x) * SMOOTHING_FACTOR)
            smooth_y = int(last_cursor_y + (target_y - last_cursor_y) * SMOOTHING_FACTOR)
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            last_cursor_x, last_cursor_y = smooth_x, smooth_y

    # ===== 6.2) Hands: detect hand landmarks & draw connections ================
    hand_results = hands.process(rgb)
    now = time.time()
    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]

        # Draw hand‐connections in red/cyan
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
            mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

        # Thumb tip & index tip for pinch distance
        lm_thumb_tip  = hand_landmarks.landmark[4]
        lm_index_tip  = hand_landmarks.landmark[8]
        lm_index_pip  = hand_landmarks.landmark[6]
        lm_middle_tip = hand_landmarks.landmark[12]
        lm_middle_pip = hand_landmarks.landmark[10]

        # Calculate pinch distance
        dist = landmark_distance(lm_thumb_tip, lm_index_tip, img_w, img_h)
        PINCH_THRESHOLD = 0.04 * img_w

        if dist < PINCH_THRESHOLD:
            # Pinch is in progress
            if not pinch_active:
                # Pinch just began
                pinch_active = True
                pinch_start_time = now
            else:
                # Pinch held: check for right-click
                if (now - pinch_start_time) >= HOLD_THRESHOLD:
                    pyautogui.click(button='right')
                    pinch_active = False
                    last_quick_pinch_time = 0
        else:
            # Pinch released
            if pinch_active:
                duration = now - pinch_start_time
                if duration < HOLD_THRESHOLD:
                    # Quick pinch: determine single vs double click
                    if (now - last_quick_pinch_time) <= DOUBLE_WINDOW:
                        pyautogui.doubleClick()
                        last_quick_pinch_time = 0
                    else:
                        pyautogui.click()
                        last_quick_pinch_time = now
                pinch_active = False

        # Two-finger scroll: index + middle extended
        index_up  = lm_index_tip.y < lm_index_pip.y
        middle_up = lm_middle_tip.y < lm_middle_pip.y
        if index_up and middle_up:
            current_y = int(lm_index_tip.y * img_h)
            if prev_scroll_y is not None:
                dy = prev_scroll_y - current_y
                # Moving fingers down → scroll up
                pyautogui.scroll(-int(dy * 2))
            prev_scroll_y = current_y
        else:
            prev_scroll_y = None

    # ===== 6.3) Display & exit on ESC ==========================================
    cv2.imshow("ControlApp", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
