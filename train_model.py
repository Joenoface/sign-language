


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