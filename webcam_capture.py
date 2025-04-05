import cv2
import numpy as np

def capture_label_from_webcam(save_path="cropped_tag.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press 'c' to capture the image (or 'q' to quit).")
    image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow('Live Feed', cv2.resize(frame, None, fx=0.5, fy=0.5))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            image = frame
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None
    cap.release()
    cv2.destroyAllWindows()

    # === Detect rectangular label ===
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                label_contour = approx

    if label_contour is None:
        print("Label not found.")
        return None

    # === Rectify and crop ===
    rect = cv2.minAreaRect(label_contour)
    angle = rect[2]
    if rect[1][1] > rect[1][0]:
        angle += 90
    if angle < -45:
        angle += 90

    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    box = cv2.boxPoints(rect)
    box = np.int0(cv2.transform(np.array([box]), M))[0]
    x, y, w, h = cv2.boundingRect(box)
    cropped = rotated[y:y+h, x:x+w]

    cv2.imwrite(save_path, cropped)
    print(f"✅ Label saved to: {save_path}")
    return save_path
