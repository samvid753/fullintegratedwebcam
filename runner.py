from pre import detect_and_crop_barcodes
from pipe import process_label_image
from webcam_capture import capture_label_from_webcam

import rethinkdb
from datetime import datetime
import os

r = rethinkdb.RethinkDB()

# Path to the raw input image
image_path = "Screenshot 2025-04-03 134728.png"
if not image_path:
    print("❌ Failed to capture label. Exiting.")
    exit()

# Step 1: Detect barcodes and crop out the label regions
cropped_images = detect_and_crop_barcodes(image_path)

# Step 2: Process label image and get JSON
json_output = process_label_image(cropped_images)

print("\n✅ All segments processed.")


# === RethinkDB Insertion ===

def setup_db(conn):
    try:
        if 'imageDB' not in r.db_list().run(conn):
            r.db_create('imageDB').run(conn)

        if 'imageTest' not in r.db('imageDB').table_list().run(conn):
            r.db('imageDB').table_create('imageTest').run(conn)
    except Exception as e:
        print(f"Error setting up DB/table: {e}")

def insert_json_and_image(conn, json_data, image_path):
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        record = {
            "name": "detection_result",
            "image": r.binary(image_bytes),
            "contentType": "image/jpeg",
            "uploadedAt": r.now(),
            "data": json_data
        }

        result = r.db("imageDB").table("imageTest").insert(record).run(conn)
        print(f"✅ Inserted into DB: {result['inserted']} success, {result['errors']} errors")

    except Exception as e:
        print(f"Error inserting data to DB: {e}")

def main():
    try:
        conn = r.connect(
            host="192.168.31.129",
            port=28015,
            user="daksh",
            password="daksh"
        )

        setup_db(conn)

        detection_image_path = "output/image_with_detections.jpg"
        insert_json_and_image(conn, json_output, detection_image_path)

    except Exception as e:
        print(f"Error connecting to RethinkDB: {e}")
    finally:
        if 'conn' in locals() and conn.is_open():
            conn.close()

if __name__ == "__main__":
    main()
