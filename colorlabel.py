import cv2
import numpy as np
import os
import json

# --- config ---
SAMPLE_FILE = "color_samples.json"
hue_tol   = 3     # ± hue tolerance
min_area  = 550   # ignore small blobs

# globals
current_frame = None
color_samples: dict[str, list[list[int]]] = {}  # label → list of [h,s,v]

def save_samples():
    """Dump color_samples to JSON."""
    with open(SAMPLE_FILE, "w") as f:
        json.dump(color_samples, f, indent=2)
    print(f"[+] Saved {sum(len(v) for v in color_samples.values())} total samples.")

def load_samples():
    """Load JSON into color_samples."""
    global color_samples
    if not os.path.isfile(SAMPLE_FILE):
        print("[ ] No sample file found, starting fresh.")
        return
    with open(SAMPLE_FILE, "r") as f:
        data = json.load(f)
    # basic sanity check
    for lbl, arr in data.items():
        if not all(isinstance(x, list) and len(x)==3 for x in arr):
            print(f"[!] Bad data for '{lbl}', skipping load.")
            return
    color_samples = data
    print(f"[+] Loaded {sum(len(v) for v in data.values())} samples for {len(data)} labels.")

def draw_palette():
    """Show one swatch per label (mean color)."""
    if not color_samples:
        return
    sw = 80
    labels = list(color_samples.keys())
    img = np.zeros((sw, sw*len(labels), 3), np.uint8)
    for i, lbl in enumerate(labels):
        hs, ss, vs = zip(*color_samples[lbl])
        mean_hsv = np.uint8([[[int(np.mean(hs)),
                               int(np.mean(ss)),
                               int(np.mean(vs))]]])
        bgr = cv2.cvtColor(mean_hsv, cv2.COLOR_HSV2BGR)[0,0]
        x0 = i*sw
        cv2.rectangle(img, (x0,0), (x0+sw,sw),
                      tuple(int(c) for c in bgr), -1)
        cv2.putText(img, lbl, (x0+5, sw-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)
    cv2.imshow("Palette", img)

def on_mouse(event, x, y, flags, param):
    """Click to sample a pixel and assign it a label."""
    global color_samples
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_frame[y, x].tolist()
        lbl = input(f"Label for HSV({h},{s},{v}): ").strip()
        if not lbl:
            print("[!] Empty label, ignored.")
            return
        color_samples.setdefault(lbl, []).append([int(h),int(s),int(v)])
        print(f"[+] Added sample under '{lbl}'")
        save_samples()
        draw_palette()

def main():
    global current_frame

    # 1) load saved samples & show palette
    load_samples()
    draw_palette()

    # 1.5) create background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    # 2) open webcam and set callback
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return
    cv2.namedWindow("Live")
    cv2.setMouseCallback("Live", on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = frame.copy()
        disp = frame.copy()

        # compute foreground mask
        fg_mask = back_sub.apply(frame)
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        )

        # for each label, build its mask and annotate blobs
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for lbl, samples in color_samples.items():
            # build mask for this label
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for (h,s,v) in samples:
                lo = np.array([max(0, h-hue_tol),  50,  50])
                hi = np.array([min(179, h+hue_tol),255,255])
                mask |= cv2.inRange(hsv, lo, hi)

            # restrict to moving/foreground pixels
            mask = cv2.bitwise_and(mask, fg_mask)

            # clean up
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

            # find and draw each blob of this color
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < min_area:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # outline + centroid + label
                cv2.drawContours(disp, [c], -1, (0,255,0), 2)
                cv2.circle(disp, (cx,cy), 5, (0,255,0), -1)
                cv2.putText(disp, lbl, (cx+8, cy-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0), 2)

            # (optionally) show this label's mask
            cv2.imshow(f"Mask: {lbl}", mask)

        # show result
        cv2.imshow("Live", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
