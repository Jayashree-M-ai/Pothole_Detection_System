# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:49:29 2026

@author: Jayashree M
"""

# -*- coding: utf-8 -*-

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles

import cv2
import os
import time
import sqlite3
import requests

from ultralytics import YOLO
from datetime import datetime

# ---------------- APP ---------------- #

app = FastAPI()

# ---------------- MODEL ---------------- #

model = None

@app.on_event("startup")
def load_model():
    global model
    model = YOLO("best.pt")

# ---------------- UPLOAD FOLDER ---------------- #

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Make uploads accessible publicly
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ---------------- TELEGRAM ---------------- #

BOT_TOKEN = "8359382623:AAFVbpzu4MI63nvNa37ddRs8HkvI3DcvCVk"
CHAT_ID = "1595313449"

def send_telegram_photo(image_path, caption):

    url = f"https://api.telegram.org/bot8359382623:AAFVbpzu4MI63nvNa37ddRs8HkvI3DcvCVk/sendPhoto"

    with open(image_path, "rb") as img:

        requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "caption": caption
            },
            files={
                "photo": img
            }
        )

def send_telegram_video(video_path, caption):

    url = f"https://api.telegram.org/bot8359382623:AAFVbpzu4MI63nvNa37ddRs8HkvI3DcvCVk/sendVideo"

    with open(video_path, "rb") as vid:

        requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "caption": caption
            },
            files={
                "video": vid
            }
        )

# ---------------- DATABASE ---------------- #

conn = sqlite3.connect(
    "potholes.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS potholes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat REAL,
    lon REAL,
    count INTEGER,
    time TEXT
)
""")

conn.commit()

def save_to_db(lat, lon, count):

    cursor.execute("""
        INSERT INTO potholes (lat, lon, count, time)
        VALUES (?, ?, ?, ?)
    """, (
        lat,
        lon,
        count,
        str(datetime.now())
    ))

    conn.commit()

# ---------------- IMAGE PROCESSING ---------------- #

def process_image(file_path, lat, lon):

    img = cv2.imread(file_path)

    results = model(img, conf=0.5)

    annotated = results[0].plot()

    pothole_count = len(results[0].boxes)

    filename = os.path.basename(file_path)

    out_filename = "output_" + filename

    out_path = os.path.join(
        UPLOAD_DIR,
        out_filename
    )

    cv2.imwrite(out_path, annotated)

    if pothole_count > 0:

        msg = f"""
🚧 Pothole Detected

Count: {pothole_count}

Time: {datetime.now()}

GPS:
{lat}, {lon}
"""

        send_telegram_photo(out_path, msg)

        save_to_db(lat, lon, pothole_count)

    return {
        "type": "image",
        "count": pothole_count,
        "output_file": f"uploads/{out_filename}"
    }

# ---------------- VIDEO PROCESSING ---------------- #
def process_video(file_path, lat, lon):

    cap = cv2.VideoCapture(file_path)

    filename = os.path.basename(file_path)

    out_filename = "output_" + filename

    out_path = os.path.join(
        UPLOAD_DIR,
        out_filename
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 20

    width = 640
    height = 360

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        out_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0

    total_potholes = 0

    last_alert_time = 0

    cooldown = 5

    # Maximum frames protection
    max_frames = 300

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Stop very long videos
        if frame_count > max_frames:
            break

        # Process only every 20th frame
        if frame_count % 20 != 0:

            resized_frame = cv2.resize(frame, (width, height))

            out.write(resized_frame)

            continue

        # Resize frame
        frame = cv2.resize(frame, (width, height))

        # YOLO prediction
        results = model(
            frame,
            conf=0.5,
            imgsz=320,
            verbose=False
        )

        annotated = results[0].plot()

        count = len(results[0].boxes)

        total_potholes += count

        current_time = time.time()

        # Telegram alert cooldown
        if count > 0 and current_time - last_alert_time > cooldown:

            temp_frame = os.path.join(
                UPLOAD_DIR,
                "temp_frame.jpg"
            )

            cv2.imwrite(temp_frame, annotated)

            msg = (
                f"🚧 Pothole Detected in Video\n\n"
                f"Count: {count}\n\n"
                f"Time: {datetime.now()}\n\n"
                f"GPS: {lat}, {lon}"
            )

            send_telegram_photo(temp_frame, msg)

            last_alert_time = current_time

        out.write(annotated)

    cap.release()

    out.release()

    # Send final video report
    if total_potholes > 0:

        send_telegram_video(
            out_path,
            f"🚧 Full Video Report\nGPS: {lat}, {lon}"
        )

        save_to_db(
            lat,
            lon,
            total_potholes
        )

    return {
        "type": "video",
        "count": total_potholes,
        "output_file": f"/uploads/{out_filename}"
    }

    





# ---------------- API ---------------- #

@app.post("/detect")

async def detect(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0)
):

    filename = file.filename.replace(" ", "_")

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(file_path, "wb") as buffer:

        buffer.write(await file.read())

    # IMAGE
    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        return process_image(
            file_path,
            lat,
            lon
        )

    # VIDEO
    elif filename.lower().endswith(
        (".mp4", ".avi", ".mov")
    ):

        return process_video(
            file_path,
            lat,
            lon
        )

    return {
        "error": "Unsupported file format"
    }

# ---------------- ROOT ---------------- #

@app.get("/")

def home():

    return {
        "message": "Pothole Detection Backend Running"
    }
