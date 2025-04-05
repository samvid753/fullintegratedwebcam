# main_pipeline.py
from pre import detect_and_crop_barcodes
from pipe import process_label_image
from webcam_capture import capture_label_from_webcam
# Path to the raw input image
image_path = "Screenshot 2025-04-03 134728.png"
if not image_path:
    print("❌ Failed to capture label. Exiting.")
    exit()


# Step 1: Detect barcodes and crop out the label regions
cropped_images = detect_and_crop_barcodes(image_path)


process_label_image(cropped_images)

print("\n✅ All segments processed.")
