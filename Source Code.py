
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import datetime
import time
from collections import Counter

# Load model
bottle_model = YOLO("C:/yolov5/runs/train/Bottles2/weights/best.pt")

# Define cocktails based on bottle combos
cocktail_recipes = {
    "Classic Vodka": ["Absolut"],
    "Gin Tonic": ["Gordons"],
    "Cuba Libre": ["Havana-Club"],
    "Irish Cream Shot": ["Baileys"],
    "Blue Lagoon": ["Skyy", "Malibu"],
    "Scotch on the Rocks": ["Scotch-Blue"],
    "Kahlua Coffee": ["Kahlua"],
    "Johnnie Ginger": ["Johnie-Walker"],
    "Tequila Sunrise": ["Jose-Cuervo"],
    "Bileys Bomb": ["Bileys"],
    "Jägerbomb": ["Jagermeister"],
    "Bombay Breeze": ["Bombay-Sapphire", "Malibu"],
    "Ballantine's Mix": ["Ballantines", "Kahlua"],
    "Captain’s Choice": ["Captain-Morgan", "Smirnoff"],
    "Tropical Storm": ["Malibu", "Skyy", "Barcadi"],
    "Whiskey Cola": ["Jim-Beam"],
    "Beefeater Lemonade": ["Beefeater"],
    "Martini Dry": ["Martini"],
    "Jose Mule": ["Jose-Cuervo"],
}

# Init video
cap = cv2.VideoCapture(0)
frame_width, frame_height = 1280, 720
cap.set(3, frame_width)
cap.set(4, frame_height)

# Settings
confidence_threshold = 0.8
cooldown_time = 3.0
bottle_counter = Counter()
bottle_last_detected = {}
per_bottle_count = Counter()
bottles_used = Counter()
cocktail_counter = Counter()

# === Ingredient Memory Buffer Setup ===
bottle_buffer = {}  # Format: {"BottleName": last_seen_timestamp}
bottle_timeout = 5  # seconds to remember a bottle after last seen

session_start_count = 0
session_end_count = 0

# Logging setup
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = open(f"log_{timestamp}.txt", "w", encoding="utf-8")

# ROI setup (centered)
roi_width, roi_height = 300, 600
roi_x1 = (frame_width - roi_width) // 2
roi_y1 = (frame_height - roi_height) // 2
roi_x2 = roi_x1 + roi_width
roi_y2 = roi_y1 + roi_height

# State control
recording_session = False
session_bottles = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (800, 800))
    frame_display = cv2.resize(frame, (frame_width, frame_height))
    height, width, _ = frame_display.shape
    scale_x = width / 800
    scale_y = height / 800
    current_time = time.time()

    # Detect bottles (with ROI and threshold)
    bottle_results = bottle_model(frame_resized, stream=False)
    for result in bottle_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            class_id = int(box.cls[0])
            class_name = bottle_model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            if not (roi_x1 <= x1 and y1 >= roi_y1 and x2 <= roi_x2 and y2 <= roi_y2):
                continue

            last_time = bottle_last_detected.get(class_name, 0)
            if current_time - last_time > cooldown_time:
                bottle_counter[class_name] += 1
                per_bottle_count[class_name] += 1
                bottle_last_detected[class_name] = current_time
                log_file.write(f"[{datetime.datetime.now()}] Detected: {class_name}\n")
                log_file.flush()

                if per_bottle_count[class_name] >= 6:
                    bottles_used[class_name] += 1
                    per_bottle_count[class_name] = 0
                    log_file.write(f"[{datetime.datetime.now()}] BOTTLE USED: {class_name}\n")
                    log_file.flush()

            if recording_session:
                session_bottles.add(class_name)

            # Draw bounding box
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_display, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw ROI
    cv2.rectangle(frame_display, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

    # Show counts
    y_offset = 20
    for i, cls in enumerate(set(bottle_counter.keys()).union(bottles_used.keys())):
        text = f"{cls}: Detections={bottle_counter[cls]}, Bottles Used={bottles_used[cls]}"
        cv2.putText(frame_display, text, (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for j, cocktail in enumerate(cocktail_counter):
        text = f"{cocktail}: {cocktail_counter[cocktail]}"
        cv2.putText(frame_display, text, (900, 30 + j * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show session start/end count
    cv2.putText(frame_display, f"Sessions Started: {session_start_count}", (10, frame_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    cv2.putText(frame_display, f"Sessions Ended: {session_end_count}", (10, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    cv2.imshow("Cocktail Tracker", frame_display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("[INFO] Session START")
        recording_session = True
        session_bottles.clear()
        session_start_count += 1

    elif key == ord('e') and recording_session:
        print("[INFO] Session END")

        current_time = time.time()
        for b in session_bottles:
            bottle_buffer[b] = current_time

        bottle_buffer = {k: v for k, v in bottle_buffer.items() if current_time - v <= bottle_timeout}
        buffered_ingredients = set(bottle_buffer.keys())

        for cocktail, required_bottles in cocktail_recipes.items():
            if set(required_bottles).issubset(buffered_ingredients):
                cocktail_counter[cocktail] += 1
                log_file.write(f"[{datetime.datetime.now()}] COCKTAIL: {cocktail}\n")
                log_file.flush()

        recording_session = False
        session_end_count += 1

    elif key == ord('q'):
        break

cap.release()

# After session ends
cap.release()
cv2.destroyAllWindows()
log_file.close()

# Write summary to separate .txt file
summary_file_path = f"summary_{timestamp}.txt"
with open(summary_file_path, "w", encoding="utf-8") as summary_file:
    summary_file.write("=== COCKTAIL SESSION SUMMARY ===\n\n")
    summary_file.write(f"Total Sessions Started: {session_start_count}\n")
    summary_file.write(f"Total Sessions Ended: {session_end_count}\n\n")

    summary_file.write("Cocktails Made:\n")
    for cocktail, count in cocktail_counter.items():
        summary_file.write(f"  {cocktail}: {count}\n")

    summary_file.write("\nBottles Used:\n")
    for bottle, count in bottles_used.items():
        summary_file.write(f"  {bottle}: {count}\n")

print(f"[INFO] Summary saved to {summary_file_path}")
cv2.destroyAllWindows()
log_file.close()
