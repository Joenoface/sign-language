
# === infer.py ===
import os
# suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import cv2, json, joblib, numpy as np
import mediapipe as mp
import tensorflow as tf

# Inference pipeline
def main_infer():
    # Load model & artifacts
    model = tf.keras.models.load_model('model/sign_language_model.keras')
    scaler = joblib.load('model/scaler.save')
    with open('model/labels.json','r') as f:
        labels = json.load(f)

    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            # flatten [x0,y0,z0, x1,y1,z1, ...]
            feats = []
            for p in lm:
                feats.extend([p.x, p.y, p.z])
            vec = np.array([feats])
            # normalize & scale
            wrist = vec[0, :3]
            vec = vec - np.repeat(wrist, vec.shape[1]//3)
            vec = scaler.transform(vec)
            # predict
            pred = model.predict(vec, verbose=0)
            idx = int(np.argmax(pred))
            sign = labels[str(idx)]
            cv2.putText(frame, sign, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Infer – press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_infer()