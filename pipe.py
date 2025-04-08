import cv2
import json
import numpy as np
import easyocr
import os
import re
from pyzbar.pyzbar import decode

from ultralytics import YOLO

def process_label_image(image_path, label_model_path="best4.pt", barcode_model_path="barcodeorignal.pt"):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    model = YOLO(label_model_path)
    results = model(image, conf=0.1, iou=0.01, imgsz=640)
    detections = results[0].boxes

    def yolo_to_pixel(x_center, y_center, w, h, img_width, img_height):
        x = max(0, int((x_center - w / 2) * img_width))
        y = max(0, int((y_center - h / 2) * img_height))
        w = max(1, min(img_width - x, int(w * img_width)))
        h = max(1, min(img_height - y, int(h * img_height)))
        return x, y, w, h

    segments = {}
    for idx, box in enumerate(detections, start=1):
        x_center, y_center, w, h = box.xywhn[0]
        bbox = yolo_to_pixel(x_center, y_center, w, h, width, height)
        segments[str(idx)] = bbox

    image_with_boxes = image.copy()
    for box in detections:
        x_center, y_center, w, h = box.xywhn[0]
        x, y, w, h = yolo_to_pixel(x_center, y_center, w, h, width, height)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, f"Segment", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_folder, "image_with_detections.jpg"), image_with_boxes)
    print("🔍 Saved image with YOLO detections to 'output/image_with_detections.jpg'")

    barcode_model = YOLO(barcode_model_path)
    reader = easyocr.Reader(['en'], gpu=False)

    detected_objects = []

    for key, bbox in segments.items():
        x, y, w, h = bbox
        segment_roi = image[y:y+h, x:x+w].copy()
        seg_height, seg_width = segment_roi.shape[:2]

        barcode_results = barcode_model(segment_roi, conf=0.1, iou=0.1)
        barcode_boxes = barcode_results[0].boxes
        decoded_barcodes = []

        for box in barcode_boxes:
            x_center, y_center, bw, bh = box.xywhn[0]
            conf = float(box.conf[0])
            bx = int((x_center - bw / 2) * seg_width)
            by = int((y_center - bh / 2) * seg_height)
            bw = int(bw * seg_width)
            bh = int(bh * seg_height)

            bx = max(0, bx)
            by = max(0, by)
            bw = min(seg_width - bx, bw)
            bh = min(seg_height - by, bh)

            barcode_roi = segment_roi[by:by+bh, bx:bx+bw]
            barcodes = decode(barcode_roi)

            for b in barcodes:
                decoded_text = b.data.decode("utf-8")
                decoded_barcodes.append(decoded_text)  # Only keep value, drop confidence

            cv2.rectangle(segment_roi, (bx, by), (bx + bw, by + bh), (255, 255, 255), -1)

        segment_filename = f"segment_{key}_barcode.jpg"
        segment_path = os.path.join(output_folder, segment_filename)
        cv2.imwrite(segment_path, segment_roi)

        results = reader.readtext(segment_path, detail=1)
        results = [r for r in results if r[2] > 0.5]

        Y_THRESHOLD = 25
        rows = []
        results.sort(key=lambda r: r[0][0][1])

        for result in results:
            bbox, text, conf = result
            y = float(bbox[0][1])
            added = False
            for row in rows:
                row_y = float(row[0][0][0][1])
                if abs(y - row_y) < Y_THRESHOLD:
                    row.append(result)
                    added = True
                    break
            if not added:
                rows.append([result])

        for row in rows:
            row.sort(key=lambda r: r[0][0][0])

        row_dict = {}
        for i, row in enumerate(rows):
            row_dict[f"Row {i+1}"] = [text for _, text, _ in row]

        print(f"\n🧾 OCR (row-wise) for segment {key}:")
        for row_key, texts in row_dict.items():
            print(f"{row_key}:", *texts)

        for (bbox, text, conf) in results:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(segment_roi, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(segment_roi, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        ocr_vis_path = os.path.join(output_folder, f"segment_{key}_ocr_visual.jpg")
        cv2.imwrite(ocr_vis_path, segment_roi)

        # Build structured key-value pairs from OCR rows (Row 1, Row 2, etc.)
        ocr_kv_pairs = {}
        for row_texts in row_dict.values():
            if len(row_texts) >= 2:
                key = " ".join(row_texts[:-1]).strip()
                value = row_texts[-1].strip()
                if key and value:
                    ocr_kv_pairs[key] = value

        detected_objects.append({
            "Barcodes": decoded_barcodes if decoded_barcodes else ["None"],
            
            "OCR KeyValue": ocr_kv_pairs
        })


    for obj in detected_objects:
        img_path = obj.get("Saved Image")
        if not img_path or not os.path.exists(img_path):
            continue

        roi = cv2.imread(img_path)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bold_mask = np.zeros(gray.shape, dtype=np.uint8)
        regular_mask = np.zeros(gray.shape, dtype=np.uint8)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > 300:
                cv2.drawContours(bold_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            else:
                cv2.drawContours(regular_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        overlay = roi.copy()
        overlay[bold_mask == 255] = (0, 0, 255)
        overlay[regular_mask == 255] = (255, 0, 0)

        alpha = 0.6
        blended = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)

        filename = obj['Segment'].replace(' ', '_') + "_blended.png"
        path = os.path.join(output_folder, filename)
        cv2.imwrite(path, blended)
        obj["Overlay Image"] = path

    reader = easyocr.Reader(['en'], gpu=False)
    final_kv_pairs = {}

    def convert_spaces(text):
        return re.sub(r' +', lambda m: r'\s' * len(m.group()), text)

    def extract_kv_from_image(image_path):
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1, upper_red1 = np.array([0, 100, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 100, 50]), np.array([180, 255, 255])
        lower_blue, upper_blue = np.array([85, 50, 50]), np.array([135, 255, 255])

        red_texts, blue_texts = [], []
        results = reader.readtext(image, detail=1)

        for (bbox, text, confidence) in results:
            if confidence < 0.5:
                continue
            processed_text = convert_spaces(text.strip()).replace('\n', r'\n')
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x1, y1 = int(top_left[0]), int(top_left[1])
            x2, y2 = int(bottom_right[0]), int(bottom_right[1])
            roi = hsv[y1:y2, x1:x2]
            mask_red = cv2.inRange(roi, lower_red1, upper_red1) | cv2.inRange(roi, lower_red2, upper_red2)
            mask_blue = cv2.inRange(roi, lower_blue, upper_blue)
            red_pixels = np.sum(mask_red > 0)
            blue_pixels = np.sum(mask_blue > 0)
            if red_pixels > blue_pixels:
                red_texts.append(processed_text)
            else:
                blue_texts.append(processed_text)

        red_text_raw = r"\n".join(red_texts).replace(r'\n', '\n')
        blue_text_raw = r"\n".join(blue_texts).replace(r'\n', '\n')
        red_lines = red_text_raw.split('\n')
        blue_lines = blue_text_raw.split('\n')

        merged_keys = []
        i = 0
        while i < len(blue_lines):
            line = blue_lines[i].replace(r'\s', ' ')
            if i + 1 < len(blue_lines) and re.fullmatch(r'\(.*\)', blue_lines[i + 1]):
                line += " " + blue_lines[i + 1].replace(r'\s', ' ')
                i += 1
            merged_keys.append(line)
            i += 1

        cleaned_values = [val.replace(r'\s', ' ') for val in red_lines]

        if not cleaned_values or all(not v.strip() for v in cleaned_values):
            if merged_keys:
                key = merged_keys[0]
                value = " ".join(merged_keys[1:]) if len(merged_keys) > 1 else ""
                return [(key, value)]
            else:
                return []
        else:
            return list(zip(merged_keys, cleaned_values[:len(merged_keys)]))

    for obj in detected_objects:
        img_path = obj.get("Overlay Image")
        if img_path and os.path.exists(img_path):
            kv_pairs = extract_kv_from_image(img_path)
            for k, v in kv_pairs:
                final_kv_pairs[k] = v

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output_json = {
        "segments": detected_objects,
        "key_value_pairs": final_kv_pairs
    }

    with open(os.path.join(output_folder, "final_key_values.json"), "w") as f:
        json.dump(convert_numpy(output_json), f, indent=4)

    print("✅ Final output saved to output/final_key_values.json")
    return output_json
