import cv2
import numpy as np

# Globals
ref_color_hsv = None
current_frame = None
tolerance = 20  # Hue tolerance in HSV

def get_roi(event, x, y, flags, param):
    """
    Mouse callback: on left-click, sample the HSV color at (x,y)
    from the latest frame.
    """
    global ref_color_hsv, current_frame
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        ref_color_hsv = hsv_frame[y, x].copy()
        print(f"Selected HSV color: {ref_color_hsv}")

def main():
    global current_frame, ref_color_hsv

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    cv2.namedWindow("Color Detection")
    # Set the mouse callback once; it will use the up-to-date current_frame
    cv2.setMouseCallback("Color Detection", get_roi)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update the global frame for the callback to use
        current_frame = frame.copy()
        display = frame.copy()

        if ref_color_hsv is not None:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Build mask around selected hue
            lower = np.array([
                max(0,   ref_color_hsv[0] - tolerance),
                50,
                50
            ])
            upper = np.array([
                min(179, ref_color_hsv[0] + tolerance),
                255,
                255
            ])
            mask = cv2.inRange(hsv, lower, upper)

            # Clean up noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:  # filter out small blobs
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Draw centroid
                        cv2.circle(display, (cx, cy), 7, (0,255,0), -1)
                        cv2.putText(display, f"({cx}, {cy})",
                                    (cx+10, cy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0,255,0), 2)
                        # Draw contour outline
                        cv2.drawContours(display, [c], -1, (0,255,0), 2)

            cv2.imshow("Mask", mask)

        cv2.imshow("Color Detection", display)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
