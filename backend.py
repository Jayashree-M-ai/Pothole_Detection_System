# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:49:29 2026

@author: Jayashree M
"""

from fastapi import FastAPI, File, UploadFile, Form
import cv2
import os
import time
import requests
from ultralytics import YOLO
from datetime import datetime

# ---------------- APP INIT ---------------- #
app = FastAPI()

model = YOLO("runs/detect/train/weights/best.pt")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- TELEGRAM CONFIG ---------------- #
BOT_TOKEN = "8359382623:AAFVbpzu4MI63nvNa37ddRs8HkvI3DcvCVk"
CHAT_ID = "1595313449"


def send_telegram_photo(image_path, caption):
    url = f"https://api.telegram.org/bot8359382623:AAFVbpzu4MI63nvNa37ddRs8HkvI3DcvCVk/sendPhoto"
    with open(image_path, "rb") as img:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": img})


def send_telegram_video(video_path, caption):
    url = f"https://api.telegram.org/bot8359382623:AAFVbpzu4MI63nvNa37ddRs8HkvI3DcvCVk/sendVideo"
    with open(video_path, "rb") as vid:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"video": vid})


# ---------------- IMAGE PROCESSING ---------------- #
def process_image(file_path, lat, lon):
    img = cv2.imread(file_path)
    results = model(img, conf=0.5)

    annotated = results[0].plot()
    pothole_count = len(results[0].boxes)

    out_path = file_path.replace(".jpg", "_out.jpg")
    cv2.imwrite(out_path, annotated)

    if pothole_count > 0:
        msg = f"""
🚧 Pothole Detected
Count: {pothole_count}
Time: {datetime.now()}
GPS: {lat}, {lon}
"""
        send_telegram_photo(out_path, msg)

    return {"type": "image", "count": pothole_count}


# ---------------- VIDEO PROCESSING ---------------- #
def process_video(file_path, lat, lon):
    cap = cv2.VideoCapture(file_path)

    out_path = file_path.replace(".mp4", "_out.mp4")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(3))
    h = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    last_alert_time = 0
    cooldown = 5  # seconds

    total_potholes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)
        annotated = results[0].plot()

        count = len(results[0].boxes)
        total_potholes += count

        current_time = time.time()

        # Send alert with cooldown
        if count > 0 and current_time - last_alert_time > cooldown:
            temp_img = file_path + "_frame.jpg"
            cv2.imwrite(temp_img, annotated)

            msg = f"""
🚧 Pothole Detected in Video
Count: {count}
Time: {datetime.now()}
GPS: {lat}, {lon}
"""
            send_telegram_photo(temp_img, msg)

            last_alert_time = current_time

        out.write(annotated)

    cap.release()
    out.release()

    if total_potholes > 0:
        send_telegram_video(out_path, f"🚧 Full Video Report | GPS: {lat}, {lon}")

    return {"type": "video", "count": total_potholes}


# ---------------- API ENDPOINT ---------------- #
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0)
):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        result = process_image(file_path, lat, lon)

    elif file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        result = process_video(file_path, lat, lon)

    else:
        return {"error": "Unsupported file format"}

    return result