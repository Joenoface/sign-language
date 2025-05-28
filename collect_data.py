# === collect_data.py ===
import os
# suppress TF C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import cv2
import mediapipe as mp
import pandas as pd

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Ensure data folder
os.makedirs('data', exist_ok=True)

# Prompt label
def main_collect():
    label = input("Enter label name: ").strip()
    if not label:
        print("Label cannot be empty.")
        return

    cap = cv2.VideoCapture(0)
    collected = []
    print("Collecting data for label: '%s'. Press 'q' to stop." % label)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                feats = []
                for lm in hand_landmarks.landmark:
                    feats.extend([lm.x, lm.y, lm.z])
                collected.append([label] + feats)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Collect Data – press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save CSV
    cols = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x","y","z")]
    df = pd.DataFrame(collected, columns=cols)
    out_path = os.path.join('data', f"{label}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == '__main__':
    main_collect()