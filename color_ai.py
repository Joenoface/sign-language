import cv2, os, json, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Config ---
SAMPLES_DIR = "samples"      # will hold subfolders per label
MODEL_FILE  = "color_model.pkl"
HIST_BINS   = [8, 8, 8]      # bins for H, S, V histograms

# Globals
current_frame = None
collect_label  = None       # when non-None, clicks save samples under this label
model          = None       # loaded/trained classifier

def ensure_dirs():
    os.makedirs(SAMPLES_DIR, exist_ok=True)

def save_patch(label, patch):
    """Save a small BGR patch under samples/label."""
    folder = os.path.join(SAMPLES_DIR, label)
    os.makedirs(folder, exist_ok=True)
    idx = len(os.listdir(folder))
    fname = os.path.join(folder, f"{idx:03d}.png")
    cv2.imwrite(fname, patch)
    print(f"[+] Saved sample {fname}")

def extract_histogram(patch):
    """Compute normalized HSV histogram feature vector."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, HIST_BINS, [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_data():
    """Load all saved patches and return X, y."""
    X, y = [], []
    for label in os.listdir(SAMPLES_DIR):
        folder = os.path.join(SAMPLES_DIR, label)
        if not os.path.isdir(folder): continue
        for imgf in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, imgf))
            if img is None: continue
            # center‐crop a 50×50 region (assuming saved patch is same)
            patch = cv2.resize(img, (50,50))
            X.append(extract_histogram(patch))
            y.append(label)
    return np.array(X), np.array(y)

def train_and_save_model():
    global model
    X, y = load_data()
    if len(y) < 2:
        print("[!] Need at least two samples of different labels to train.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    clf = SVC(kernel="rbf", probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("--- Training complete. Test set report: ---")
    print(classification_report(y_test, y_pred))
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    model = clf
    print(f"[+] Model saved to {MODEL_FILE}")

def load_model():
    global model
    if not os.path.isfile(MODEL_FILE):
        print("[ ] No model file found. Run training first (press 't').")
        return
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"[+] Loaded model from {MODEL_FILE}")

def on_mouse(event, x, y, flags, param):
    global collect_label, current_frame
    if event == cv2.EVENT_LBUTTONDOWN and collect_label and current_frame is not None:
        # take 50×50 patch centered at click
        h, w = current_frame.shape[:2]
        x0 = max(0, x-25); x1 = min(w, x+25)
        y0 = max(0, y-25); y1 = min(h, y+25)
        patch = current_frame[y0:y1, x0:x1]
        save_patch(collect_label, patch)

def main():
    global current_frame, collect_label, model

    ensure_dirs()
    load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam."); return

    cv2.namedWindow("Live")
    cv2.setMouseCallback("Live", on_mouse)

    print("""
Controls:
  c → enter **collect** mode: terminal asks for a label name.
        Then click on the video to save 50×50 patches under that label.
        Press ENTER (empty) to exit collect mode.
  t → train model on all saved data (requires ≥2 labels).
  l → load last-saved model.
  q → quit.
""")

    while True:
        ret, frame = cap.read()
        if not ret: break
        current_frame = frame.copy()
        disp = frame.copy()

        # if model loaded, segment by simple color blob detection + classification
        if model:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # sliding window-ish: scan with stride on a coarse grid
            step = 40
            for y in range(25, frame.shape[0], step):
                for x in range(25, frame.shape[1], step):
                    patch = frame[y-25:y+25, x-25:x+25]
                    if patch.shape[0]!=50 or patch.shape[1]!=50: continue
                    feat = extract_histogram(patch).reshape(1,-1)
                    label = model.predict(feat)[0]
                    prob  = model.predict_proba(feat).max()
                    if prob > 0.7:  # confidence threshold
                        cv2.rectangle(disp, (x-25,y-25), (x+25,y+25), (0,255,0), 1)
                        cv2.putText(disp, label, (x-25, y-30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Live", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            collect_label = input("Collect mode — enter label (empty to cancel): ").strip() or None
            if collect_label:
                print(f"[+] Now collecting samples for label: '{collect_label}'. Click to save.")
            else:
                print("[ ] Exiting collect mode.")
        elif key == ord('t'):
            train_and_save_model()
        elif key == ord('l'):
            load_model()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
