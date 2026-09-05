"""
utils.py
Helper functions for the Face & Emotion Analytics Dashboard:
- thread-safe CSV logging (webcam runs in a separate WebRTC thread)
- analytics aggregation for the dashboard charts
- drawing helpers for the video overlay
"""

import os
import csv
import time
import threading
from datetime import datetime

import cv2
import pandas as pd

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "emotion_log.csv")
LOG_COLUMNS = ["timestamp", "dominant_emotion", "angry", "disgust", "fear",
               "happy", "sad", "surprise", "neutral", "face_count"]

_lock = threading.Lock()
_last_write_ts = 0.0
MIN_WRITE_INTERVAL = 1.0  # seconds between log rows, keeps the CSV from exploding


def ensure_log_file():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)


def log_emotion(dominant_emotion: str, scores: dict, face_count: int, force: bool = False):
    """Thread-safe, rate-limited append to the CSV log."""
    global _last_write_ts
    now = time.time()
    with _lock:
        if not force and (now - _last_write_ts) < MIN_WRITE_INTERVAL:
            return
        _last_write_ts = now
        ensure_log_file()
        row = [
            datetime.now().isoformat(timespec="seconds"),
            dominant_emotion,
            round(scores.get("angry", 0), 2),
            round(scores.get("disgust", 0), 2),
            round(scores.get("fear", 0), 2),
            round(scores.get("happy", 0), 2),
            round(scores.get("sad", 0), 2),
            round(scores.get("surprise", 0), 2),
            round(scores.get("neutral", 0), 2),
            face_count,
        ]
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow(row)


def load_log() -> pd.DataFrame:
    ensure_log_file()
    try:
        df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    except Exception:
        df = pd.DataFrame(columns=LOG_COLUMNS)
        return df
    return df


def clear_log():
    with _lock:
        ensure_log_file()
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)


def session_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total": 0, "top_emotion": "-", "duration_min": 0, "avg_faces": 0}
    duration_min = 0
    if len(df) > 1:
        duration_min = round(
            (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60, 1
        )
    return {
        "total": len(df),
        "top_emotion": df["dominant_emotion"].mode().iloc[0] if not df["dominant_emotion"].mode().empty else "-",
        "duration_min": duration_min,
        "avg_faces": round(df["face_count"].mean(), 2),
    }


EMOTION_COLORS = {
    "angry": "#E4572E",
    "disgust": "#6A994E",
    "fear": "#7C4585",
    "happy": "#F4A300",
    "sad": "#3D5A80",
    "surprise": "#EE964B",
    "neutral": "#8D99AE",
}


def draw_face_box(frame, x, y, w, h, label, color=(255, 189, 89)):
    """Draw a styled bounding box + label on a BGR frame (OpenCV)."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return frame
