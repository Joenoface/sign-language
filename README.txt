# Real-Time Sign Language Recognition

This project implements a complete pipeline for real-time sign language recognition using MediaPipe for hand landmark detection and TensorFlow Keras for model training and inference. It consists of three main scripts:

1. **`collect_data.py`** — Capture and label hand landmark data via webcam.
2. **`train_model.py`** — Preprocess data, train a neural network classifier, and save model artifacts.
3. **`infer.py`** — Load the trained model and perform real-time sign prediction on webcam input.

---

## 📁 Repository Structure
```
sign-language/
├─ data/                   # Collected CSV files per sign label
├─ model/                  # Saved model, scaler, and label map
│  ├─ best_model.h5
│  ├─ sign_language_model.h5
│  ├─ scaler.save
│  └─ labels.json
├─ logs/                   # TensorBoard logs
├─ collect_data.py         # Data collection script
├─ train_model.py          # Training & evaluation script
├─ infer.py                # Real-time inference script
├─ .venv/                  # Python virtual environment
└─ README.md               # This file
```

---

## ⚙️ Setup

1. **Clone the repository** and navigate into it:
   ```bash
   git clone <your-repo-url>
   cd sign-language
   ```

2. **Create & activate** the virtual environment (Windows PowerShell):
   ```powershell
   python -m venv .venv --upgrade-deps
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install opencv-python mediapipe pandas scikit-learn joblib tensorflow
   ```

4. **Configure VS Code** (optional):
   - Select the interpreter at `./.venv/Scripts/python.exe` in **Workspace** settings.
   - Enable automatic terminal activation.

---

## 🖥️ Usage

### 1. Collect Data

Run `collect_data.py` to capture hand landmarks for each sign.
```bash
python collect_data.py
```
- Enter a label (e.g. `hello`, `thank_you`, etc.).
- Show the sign in front of the webcam. Press **q** to stop.
- A CSV file `data/<label>.csv` will be created.
- Repeat for all desired signs.

### 2. Train Model

Once you have CSVs for each sign, train the classifier:
```bash
python train_model.py
```
- Normalizes and standardizes data.
- Splits into train/test sets.
- Trains a 3-layer dense neural network with callbacks.
- Saves:
  - Best weights: `model/best_model.h5`
  - Final model: `model/sign_language_model.h5` (HDF5) and `.keras` (optional)
  - Scaler: `model/scaler.save`
  - Label map: `model/labels.json`
- Prints test accuracy, confusion matrix, and classification report.

To view training curves:
```bash
tensorboard --logdir logs
```
Then open http://localhost:6006 in a browser.

### 3. Real-Time Inference

Run `infer.py` to see live predictions:
```bash
python infer.py
```
- Overlays predicted sign on the webcam feed.
- Press **q** to quit.

---

## 🔧 Customization & Tips

- **Add new signs**: Collect additional CSVs via `collect_data.py` and retrain.
- **Improve robustness**: Gather data under varied lighting, backgrounds, and from multiple users.
- **Hyperparameter tuning**: Adjust network architecture, batch size, epochs, and callbacks in `train_model.py`.
- **Export format**: Use `model.save('model/sign_language_model.keras')` for native Keras format.

---

## ⚖️ License

This project is released under the MIT License. See `LICENSE` for details.

