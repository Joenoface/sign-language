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


# === train_model.py ===
import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Train pipeline
def main_train():
    # Load CSVs
    X_list, y_list = [], []
    label_map = {}
    for idx, fname in enumerate(sorted(os.listdir('data'))):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join('data', fname))
            X_list.append(df.iloc[:,1:].values)
            y_list.append(np.full(len(df), idx))
            label_map[idx] = fname.replace('.csv','')
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Save label map
    os.makedirs('model', exist_ok=True)
    with open('model/labels.json','w') as f:
        json.dump(label_map, f)

    # Normalize & standardize
    wrist = X[:, :3]
    X = X - np.repeat(wrist, X.shape[1]//3, axis=1)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    joblib.dump(scaler, 'model/scaler.save')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build model
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('model/best_model.h5', save_best_only=True, monitor='val_accuracy'),
        TensorBoard(log_dir='logs')
    ]

    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50, batch_size=32,
        callbacks=callbacks
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save final
    model.save('model/sign_language_model.keras')
    print("Model saved to model/sign_language_model.keras")

if __name__ == '__main__':
    main_train()


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
