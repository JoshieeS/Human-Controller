# controller.py

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# ─── 1) Initialize MediaPipe modules ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,       # gives iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils  # for debug overlays

# ─── 2) Get screen size ───────────────────────────────────────────────────────
screen_w, screen_h = pyautogui.size()

# Move mouse to center at startup
pyautogui.moveTo(screen_w // 2, screen_h // 2, duration=0)

# ─── 3) Helper: distance between two landmarks ────────────────────────────────
def landmark_distance(lm1, lm2, img_w, img_h):
    x1, y1 = int(lm1.x * img_w), int(lm1.y * img_h)
    x2, y2 = int(lm2.x * img_w), int(lm2.y * img_h)
    return np.hypot(x2 - x1, y2 - y1)

# ─── 4) Open webcam ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
# Optional: reduce resolution if performance is an issue
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_click = False
click_cooldown = 0.5
last_click_time = time.time()
prev_scroll_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror horizontally
    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ----- 4.1) FaceMesh: get iris centers for gaze cursor  --------------------
    face_results = face_mesh.process(rgb)
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]

        # Debug: draw face mesh (optional)
        mp_draw.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
        )

        # Iris landmark indices (refined mesh)
        LEFT_IRIS  = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        # Compute normalized center of left iris
        lx = np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS])
        ly = np.mean([face_landmarks.landmark[i].y for i in LEFT_IRIS])
        # Compute normalized center of right iris
        rx = np.mean([face_landmarks.landmark[i].x for i in RIGHT_IRIS])
        ry = np.mean([face_landmarks.landmark[i].y for i in RIGHT_IRIS])

        # Average between left/right to get overall gaze point
        gaze_x = (lx + rx) / 2
        gaze_y = (ly + ry) / 2

        # Amplify range so eyes looking fully left/right maps to screen edges.
        # We increase the factor to 3 for more drastic cursor movement.
        factor = 5.0
        def stretch(v, factor):
            return max(0.0, min(1.0, (v - 0.5) * factor + 0.5))

        nx = stretch(gaze_x, factor=factor)
        ny = stretch(gaze_y, factor=factor)

        # Map normalized to screen coords
        cursor_x = int(nx * screen_w)
        cursor_y = int(ny * screen_h)

        # Move mouse instantly to gaze point
        pyautogui.moveTo(cursor_x, cursor_y, duration=0)

    # ----- 4.2) Hands: pinch-click and scroll  ----------------------------------
    hand_results = hands.process(rgb)
    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]

        # Debug: draw hand landmarks (optional)
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
            mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

        # Thumb tip and index tip for pinch
        lm_thumb_tip  = hand_landmarks.landmark[4]
        lm_index_tip  = hand_landmarks.landmark[8]
        lm_index_pip  = hand_landmarks.landmark[6]
        lm_middle_tip = hand_landmarks.landmark[12]
        lm_middle_pip = hand_landmarks.landmark[10]

        # ---- 1) Pinch = left-click ----
        dist = landmark_distance(lm_thumb_tip, lm_index_tip, img_w, img_h)
        PINCH_THRESHOLD = 0.04 * img_w  # adjust based on your camera setup
        now = time.time()
        if dist < PINCH_THRESHOLD:
            if not prev_click and (now - last_click_time) > click_cooldown:
                pyautogui.click()
                last_click_time = now
                prev_click = True
        else:
            prev_click = False

        # ---- 2) Two-finger scroll (index+middle up) ----
        index_up  = lm_index_tip.y < lm_index_pip.y
        middle_up = lm_middle_tip.y < lm_middle_pip.y
        if index_up and middle_up:
            current_y = int(lm_index_tip.y * img_h)
            if prev_scroll_y is not None:
                dy = prev_scroll_y - current_y
                # Invert direction: moving fingers down → scroll up
                pyautogui.scroll(-int(dy * 2))
            prev_scroll_y = current_y
        else:
            prev_scroll_y = None

    # ----- 4.3) (Removed exit gesture) Only ESC key to quit --------------------

    # ----- 5) Display debug window -----
    cv2.imshow("ControlApp", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
