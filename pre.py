import cv2
import numpy as np
from ultralytics import YOLO
import os
def detect_and_crop_barcodes(image_path, barcode_model_path="barcode.pt", border=100, margin=0):
   

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Load image
   
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Load YOLO model
    model = YOLO(barcode_model_path)

    # Run detection
    results = model(image, conf=0.7, iou=0.2)
    detections = results[0].boxes

    margin = 0    # margin around detected object
    thick_border = 100   # black border thickness

    for idx, box in enumerate(detections, start=1):
        x_center, y_center, w, h = box.xywhn[0]

        # Convert normalized to pixel values
        x = int((x_center - w / 2) * width)
        y = int((y_center - h / 2) * height)
        w = int(w * width)
        h = int(h * height)

        # Apply margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)

        # Extract only the detected part with margin
        extracted = image[y1:y2, x1:x2].copy()

        # Add black border around the extracted object
        final_image = cv2.copyMakeBorder(
            extracted,
            top=thick_border,
            bottom=thick_border,
            left=thick_border,
            right=thick_border,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Save the result
        output_path = os.path.join(output_folder, f"only_detected_{idx}.jpg")
        cv2.imwrite(output_path, final_image)
        return output_path
    
    # Done — no full image saving because you only want the detected parts
if __name__ == "__main__":
    detect_and_crop_barcodes("path/to/your/input.jpg")
