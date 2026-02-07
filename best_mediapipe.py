import cv2
import mediapipe as mp
import time
import pickle
import pandas as pd
import os
import numpy as np

PILOT_MAP = {0: "DOWN", 1: "HOVER", 2: "LAND", 3: "LEFT", 4: "RIGHT", 5: "UP"}
CONFIRMATION_TIME_UTIL = 1.0
CONFIRMATION_TIME_MAIN = 0.5

with open("gesture_model_final.p", "rb") as f:
    data = pickle.load(f)
    if type(data) is dict:
        model = data["model"]
    else:
        model = data

with open("gesture_model_final_util.p", "rb") as f:
    model_util = pickle.load(f)

feature_cols = ["is_right_hand"]

for i in range(21):
    feature_cols.extend([f"x{i}", f"y{i}", f"z{i}"])

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
)
handlandmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing_styles = mp.tasks.vision.drawing_styles

pending_main_gesture = "HOVER"
locked_main_gesture = "HOVER"
current_util_gesture = "NONE"

util_gesture_start_time = 0
main_gesture_start_time = 0

fence_mode = "OFF"
drone_speed = "SLOW"

cam = cv2.VideoCapture(1)

print("-----STARTING-----")

while True:
    success, img = cam.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    results = handlandmarker.detect_for_video(mp_image, int(time.time() * 1000))

    right_hand_data = None
    left_hand_data = None

    locked_main_gesture = "HOVER"

    if results.hand_landmarks:
        for i, hand_lms in enumerate(results.hand_landmarks):
            mp_drawing.draw_landmarks(
                img,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            is_right_hand = 1 if results.handedness[i][0].category_name == "Left" else 0

            pred_row = [is_right_hand]
            for point in hand_lms:
                pred_row.extend([point.x, point.y, point.z])
            X = pd.DataFrame([pred_row], columns=feature_cols)
            wrist_pos = (int(hand_lms[0].x * w), int(hand_lms[0].y * h))

            if is_right_hand == 1:
                right_hand_data = {"X": X, "wrist": wrist_pos, "lms": hand_lms}
            else:
                left_hand_data = {"X": X, "wrist": wrist_pos}

    final_util_cmd = "HOVER"

    if left_hand_data and right_hand_data:
        probs_util = model_util.predict_proba(left_hand_data["X"])[0]
        if max(probs_util) > 0.6:
            final_util_cmd = model_util.classes_[probs_util.argmax()]

        if final_util_cmd != current_util_gesture:
            current_util_gesture = final_util_cmd
            util_gesture_start_time = time.time()

        progress = min(
            (time.time() - util_gesture_start_time) / CONFIRMATION_TIME_UTIL, 1.0
        )

        if final_util_cmd in ["TRACK", "REPLAY", "SPEED", "FLIP"]:
            cv2.ellipse(
                img,
                left_hand_data["wrist"],
                (30, 30),
                0,
                0,
                360 * progress,
                (0, 255, 255),
                4,
            )
            cv2.putText(
                img,
                final_util_cmd,
                (left_hand_data["wrist"][0] - 20, left_hand_data["wrist"][1] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        if progress >= 1.0:
            if final_util_cmd == "TRACK":
                fence_mode = "FOLLOW_ME"
            elif final_util_cmd == "REPLAY":
                fence_mode = "OFF"
            elif final_util_cmd == "SPEED":
                if drone_speed == "FAST":
                    drone_speed = "SLOW"
                else:
                    drone_speed = "FAST"
            elif final_util_cmd == "FLIP":
                pass
            util_gesture_start_time = time.time()

    if right_hand_data:
        hand_px = (
            int(right_hand_data["lms"][8].x * w),
            int(right_hand_data["lms"][8].y * h),
        )

        if fence_mode == "FOLLOW_ME":
            center_x, center_y = w // 2, h // 2
            dead_zone = 60

            cv2.circle(img, (center_x, center_y), dead_zone, (255, 255, 255), 1)
            cv2.line(img, (center_x, center_y), hand_px, (0, 255, 255), 2)

            error_x = hand_px[0] - center_x
            error_y = center_y - hand_px[1]
            dist = (error_x**2 + error_y**2) ** 0.5

            if dist < dead_zone:
                locked_main_gesture = "HOVER"
                cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), -1)
            else:
                if abs(error_x) > abs(error_y):
                    locked_main_gesture = "RIGHT" if error_x > 0 else "LEFT"
                else:
                    locked_main_gesture = "UP" if error_y > 0 else "DOWN"

        else:
            probs_main = model.predict_proba(right_hand_data["X"])[0]
            raw_main = pending_main_gesture

            if max(probs_main) > 0.6:
                idx = probs_main.argmax()
                if idx in PILOT_MAP:
                    raw_main = PILOT_MAP[idx]

            if raw_main != pending_main_gesture:
                pending_main_gesture = raw_main
                main_gesture_start_time = time.time()

            progress = min(
                (time.time() - main_gesture_start_time) / CONFIRMATION_TIME_MAIN, 1.0
            )

            if pending_main_gesture != locked_main_gesture:
                cv2.ellipse(
                    img,
                    right_hand_data["wrist"],
                    (30, 30),
                    0,
                    0,
                    360 * progress,
                    (0, 255, 0),
                    4,
                )
                cv2.putText(
                    img,
                    pending_main_gesture,
                    (
                        right_hand_data["wrist"][0] - 20,
                        right_hand_data["wrist"][1] - 40,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if progress >= 1.0:
                locked_main_gesture = pending_main_gesture

    elif left_hand_data:
        probs_main = model.predict_proba(left_hand_data["X"])[0]
        raw_main = pending_main_gesture

        if max(probs_main) > 0.6:
            idx = probs_main.argmax()
            if idx in PILOT_MAP:
                raw_main = PILOT_MAP[idx]

        if raw_main != pending_main_gesture:
            pending_main_gesture = raw_main
            main_gesture_start_time = time.time()

        progress = min(
            (time.time() - main_gesture_start_time) / CONFIRMATION_TIME_MAIN, 1.0
        )

        if pending_main_gesture != locked_main_gesture:
            cv2.ellipse(
                img,
                left_hand_data["wrist"],
                (30, 30),
                0,
                0,
                360 * progress,
                (0, 255, 255),
                4,
            )
            cv2.putText(
                img,
                f"BACKUP: {pending_main_gesture}",
                (left_hand_data["wrist"][0], left_hand_data["wrist"][1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        if progress >= 1.0:
            locked_main_gesture = pending_main_gesture

    cv2.rectangle(img, (0, 0), (640, 60), (0, 0, 0), -1)

    cv2.putText(
        img,
        f"CMD: {locked_main_gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    spd_color = (0, 255, 0) if drone_speed == "SLOW" else (0, 255, 255)
    cv2.putText(
        img,
        f"SPD: {drone_speed}",
        (350, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        spd_color,
        2,
    )

    mode_color = (0, 255, 255) if fence_mode == "OFF" else (0, 255, 0)
    cv2.putText(
        img,
        f"MODE: {fence_mode}",
        (500, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        mode_color,
        2,
    )

    cv2.imshow("Drone Vision", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
