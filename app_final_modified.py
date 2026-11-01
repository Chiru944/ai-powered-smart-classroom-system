from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request,
    redirect,
    session,
    send_file,
    url_for,
)
import base64
import io
import json
import os
import shutil
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


# ================= APP =================

app = Flask(__name__)
app.secret_key = "secret123"


# ================= LOGIN =================

USERNAME = "admin"
PASSWORD = "1234"


# ================= TEST / PRODUCTION SETTINGS =================
# Initial testing:
LAST_MINUTES_FOR_ATTENDANCE = 20
MIN_RECOGNITION_COUNT = 3

# After testing, change to:
# LAST_MINUTES_FOR_ATTENDANCE = 10
# MIN_RECOGNITION_COUNT = 3

RECOGNITION_INTERVAL_SECONDS = 5
LBPH_CONFIDENCE_THRESHOLD = 70

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(BASE_DIR, "database.db")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.json")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REGISTRATION_TEMP_FOLDER = os.path.join(BASE_DIR, "registration_temp")

REGISTRATION_REQUIRED_IMAGES = 10
REGISTRATION_MIN_FACE_SIZE = 120
REGISTRATION_MIN_BRIGHTNESS = 45
REGISTRATION_MAX_BRIGHTNESS = 220
REGISTRATION_MIN_BLUR_SCORE = 55.0

os.makedirs(TRAINER_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(REGISTRATION_TEMP_FOLDER, exist_ok=True)


# ================= MEDIAPIPE =================

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5,
)


# ================= HAAR CASCADES =================

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

attendance_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ================= GLOBAL CLASSROOM STATS =================

stats = {
    "students": 0,
    "attentive": 0,
    "distracted": 0,
    "engagement": 0,
}

last_saved_time = 0
session_id = datetime.now().strftime("%Y%m%d_%H")
sleep_timers = {}


# ================= GLOBAL ATTENDANCE STATE =================

attendance_lock = threading.Lock()

lbph_recognizer = None
lbph_label_map = {}

active_attendance_session = None
recognition_counts = defaultdict(int)
last_recognition_times = {}

last_active_subject = None
last_active_session_key = None
last_active_class_end = None

finalized_sessions = set()


# ================= GLOBAL REGISTRATION STATE =================

registration_lock = threading.Lock()
registration_state = {
    "active": False,
    "usn": None,
    "name": None,
    "edit_mode": False,
    "count": 0,
    "images": [],
}


# ================= DATABASE INITIALIZATION =================

def init_db():
    conn = None

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS students(
                usn TEXT PRIMARY KEY,
                name TEXT,
                photo BLOB
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS student_images(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usn TEXT NOT NULL,
                image_number INTEGER NOT NULL,
                photo BLOB NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(usn, image_number),
                FOREIGN KEY(usn) REFERENCES students(usn)
                    ON DELETE CASCADE
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS subjects(
                subject_code TEXT PRIMARY KEY,
                subject_name TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS timetable(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_code TEXT,
                day TEXT,
                start_time TEXT,
                end_time TEXT,
                period INTEGER
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS attendance(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usn TEXT,
                subject_code TEXT,
                date TEXT,
                period INTEGER,
                status TEXT,
                entry_time TEXT,
                recognition_count INTEGER DEFAULT 0
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT,
                date TEXT,
                time TEXT,
                subject_code TEXT,
                period INTEGER,
                students INTEGER,
                attentive INTEGER,
                distracted INTEGER,
                engagement INTEGER
            )
        """)

        conn.commit()

    except sqlite3.Error as error:
        if conn is not None:
            conn.rollback()
        print("Database initialization error:", error)

    finally:
        if conn is not None:
            conn.close()


def migrate_attendance_table():
    """
    Safely adds missing attendance columns and creates a unique index.
    Existing data is preserved. Duplicate rows are reduced to one row
    per USN, subject, date and period before the index is created.
    """
    conn = None

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        c.execute("PRAGMA table_info(attendance)")
        existing_columns = {row[1] for row in c.fetchall()}

        if "entry_time" not in existing_columns:
            c.execute("""
                ALTER TABLE attendance
                ADD COLUMN entry_time TEXT
            """)

        if "recognition_count" not in existing_columns:
            c.execute("""
                ALTER TABLE attendance
                ADD COLUMN recognition_count INTEGER DEFAULT 0
            """)

        c.execute("""
            DELETE FROM attendance
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM attendance
                GROUP BY usn, subject_code, date, period
            )
        """)

        c.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_unique_attendance_class
            ON attendance(usn, subject_code, date, period)
        """)

        conn.commit()
        print("Attendance table migration completed")

    except sqlite3.Error as error:
        if conn is not None:
            conn.rollback()
        print("Attendance migration error:", error)

    finally:
        if conn is not None:
            conn.close()


def insert_default_subjects():
    conn = None

    subjects = [
        ("1BMATCS201", "Mathematics"),
        ("1BPHYCS202", "Physics"),
        ("1BESC204B", "ESC-II"),
        ("1BPIC205", "Programming in C"),
        ("1BSS207", "Soft Skills"),
        ("1BSK209", "Sanskrit/Kannada"),
        ("1BCEDCS203", "CAED"),
        ("1BPICL206", "PSC Lab"),
        ("1BPRJ208", "IDP"),
    ]

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        c.executemany("""
            INSERT OR IGNORE INTO subjects(subject_code, subject_name)
            VALUES (?, ?)
        """, subjects)

        conn.commit()

    except sqlite3.Error as error:
        if conn is not None:
            conn.rollback()
        print("Default-subject insertion error:", error)

    finally:
        if conn is not None:
            conn.close()


# ================= TIMETABLE HELPERS =================

def get_current_subject(now=None):
    if now is None:
        now = datetime.now()

    conn = None

    try:
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M:%S")

        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT *
            FROM timetable
            WHERE day = ?
              AND time(start_time) <= time(?)
              AND time(end_time) >= time(?)
            ORDER BY time(start_time)
            LIMIT 1
        """, (
            current_day,
            current_time,
            current_time,
        ))

        return c.fetchone()

    except sqlite3.Error as error:
        print("Current-subject lookup error:", error)
        return None

    finally:
        if conn is not None:
            conn.close()


def parse_timetable_datetime(date_value, time_value):
    time_text = str(time_value).strip()

    for time_format in ("%H:%M:%S", "%H:%M"):
        try:
            parsed_time = datetime.strptime(time_text, time_format).time()
            return datetime.combine(date_value, parsed_time)
        except ValueError:
            continue

    raise ValueError(f"Invalid timetable time: {time_text}")


def get_class_times(subject_row, now=None):
    if now is None:
        now = datetime.now()

    class_start = parse_timetable_datetime(
        now.date(),
        subject_row["start_time"],
    )

    class_end = parse_timetable_datetime(
        now.date(),
        subject_row["end_time"],
    )

    attendance_start = class_end - timedelta(
        minutes=LAST_MINUTES_FOR_ATTENDANCE
    )

    return class_start, attendance_start, class_end


def is_attendance_window_active(subject_row, now=None):
    if now is None:
        now = datetime.now()

    try:
        _, attendance_start, class_end = get_class_times(subject_row, now)
        return attendance_start <= now <= class_end
    except (ValueError, TypeError, KeyError) as error:
        print("Attendance-window calculation error:", error)
        return False


def has_class_ended(class_end, now=None):
    if now is None:
        now = datetime.now()

    return class_end is not None and now >= class_end


def build_attendance_session_key(subject_row, now=None):
    if now is None:
        now = datetime.now()

    return (
        f"{now.strftime('%Y-%m-%d')}|"
        f"{subject_row['subject_code']}|"
        f"{subject_row['period']}"
    )


# ================= LBPH MODEL AND LABEL MAP =================

def load_label_map():
    global lbph_label_map

    if not os.path.exists(LABELS_PATH):
        print(f"Label map missing: {LABELS_PATH}")
        lbph_label_map = {}
        return False

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
            loaded_map = json.load(labels_file)

        cleaned_map = {}

        for label_text, student_data in loaded_map.items():
            label_id = int(label_text)

            if (
                isinstance(student_data, dict)
                and student_data.get("usn")
                and student_data.get("name")
            ):
                cleaned_map[label_id] = {
                    "usn": str(student_data["usn"]),
                    "name": str(student_data["name"]),
                }

        lbph_label_map = cleaned_map

        if not lbph_label_map:
            print("labels.json exists but contains no valid mappings")
            return False

        print(f"Loaded {len(lbph_label_map)} LBPH label mappings")
        return True

    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print("Label-map loading error:", error)
        lbph_label_map = {}
        return False


def load_lbph_model():
    global lbph_recognizer

    if not hasattr(cv2, "face"):
        print(
            "OpenCV LBPH is unavailable. Install opencv-contrib-python."
        )
        lbph_recognizer = None
        return False

    if not os.path.exists(TRAINER_PATH):
        print(f"Trainer model missing: {TRAINER_PATH}")
        lbph_recognizer = None
        return False

    if not load_label_map():
        print("LBPH model not activated because labels.json is unavailable")
        lbph_recognizer = None
        return False

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)
        lbph_recognizer = recognizer
        print("LBPH trainer loaded successfully")
        return True

    except cv2.error as error:
        print("LBPH trainer loading error:", error)
        lbph_recognizer = None
        return False



# ================= AUTOMATIC FACE REGISTRATION =================

def _reset_registration_state():
    with registration_lock:
        registration_state.update({
            "active": False,
            "usn": None,
            "name": None,
            "edit_mode": False,
            "count": 0,
            "images": [],
        })


def _decode_data_url(image_value):
    if not image_value or "," not in image_value:
        raise ValueError("A valid webcam image is required")

    encoded = image_value.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("The webcam image could not be decoded")

    return frame


def _validate_and_prepare_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = attendance_face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=6,
        minSize=(REGISTRATION_MIN_FACE_SIZE, REGISTRATION_MIN_FACE_SIZE),
    )

    if len(faces) == 0:
        return None, None, "No clear face detected. Move closer and look at the camera."

    if len(faces) > 1:
        return None, None, "More than one face detected. Keep only one student visible."

    x, y, w, h = faces[0]
    face_gray = gray[y:y + h, x:x + w]
    brightness = float(np.mean(face_gray))
    blur_score = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())

    if brightness < REGISTRATION_MIN_BRIGHTNESS:
        return None, None, "Image is too dark. Improve the lighting."

    if brightness > REGISTRATION_MAX_BRIGHTNESS:
        return None, None, "Image is too bright. Reduce direct light on the face."

    if blur_score < REGISTRATION_MIN_BLUR_SCORE:
        return None, None, "Image is blurry. Keep your head and camera steady."

    margin_x = int(w * 0.12)
    margin_y = int(h * 0.15)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + w + margin_x)
    y2 = min(frame.shape[0], y + h + margin_y)

    color_crop = frame[y1:y2, x1:x2]
    training_face = cv2.resize(face_gray, (200, 200))
    training_face = cv2.equalizeHist(training_face)

    ok_color, color_bytes = cv2.imencode(
        ".jpg", color_crop, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    ok_gray, gray_bytes = cv2.imencode(
        ".jpg", training_face, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if not ok_color or not ok_gray:
        return None, None, "The face image could not be prepared. Please try again."

    return color_bytes.tobytes(), gray_bytes.tobytes(), "Face accepted"


def train_all_registered_faces():
    global lbph_recognizer, lbph_label_map

    if not hasattr(cv2, "face"):
        return False, "OpenCV LBPH is unavailable. Install opencv-contrib-python."

    os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        students = conn.execute(
            "SELECT usn, name FROM students ORDER BY usn"
        ).fetchall()

        faces = []
        labels = []
        label_map = {}

        for label_id, student in enumerate(students):
            usn = str(student["usn"])
            name = str(student["name"])
            student_dir = os.path.join(DATASET_DIR, usn)
            image_paths = []

            if os.path.isdir(student_dir):
                image_paths.extend(sorted(
                    str(path) for path in Path(student_dir).glob("*.jpg")
                ))

            legacy_path = os.path.join(DATASET_DIR, f"{usn}.jpg")
            if os.path.exists(legacy_path):
                image_paths.append(legacy_path)

            student_face_count = 0

            for image_path in image_paths:
                gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue

                detected = attendance_face_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
                )

                if len(detected) > 0:
                    x, y, w, h = max(detected, key=lambda box: box[2] * box[3])
                    gray = gray[y:y + h, x:x + w]

                if gray.size == 0:
                    continue

                gray = cv2.resize(gray, (200, 200))
                gray = cv2.equalizeHist(gray)
                faces.append(gray)
                labels.append(label_id)
                student_face_count += 1

            if student_face_count:
                label_map[label_id] = {"usn": usn, "name": name}

        if not faces:
            lbph_recognizer = None
            lbph_label_map = {}
            return False, "No usable registered face images were found."

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels, dtype=np.int32))
        recognizer.write(TRAINER_PATH)

        with open(LABELS_PATH, "w", encoding="utf-8") as labels_file:
            json.dump(label_map, labels_file, indent=2)

        lbph_recognizer = recognizer
        lbph_label_map = label_map

        return True, (
            f"Face model trained using {len(faces)} images "
            f"for {len(label_map)} students."
        )

    except (sqlite3.Error, OSError, cv2.error, ValueError) as error:
        return False, f"Face model training error: {error}"

    finally:
        conn.close()


# ================= ATTENDANCE SESSION MANAGEMENT =================

def reset_attendance_session(session_key=None):
    global active_attendance_session
    global recognition_counts
    global last_recognition_times

    with attendance_lock:
        active_attendance_session = session_key
        recognition_counts = defaultdict(int)
        last_recognition_times = {}

    if session_key:
        print("Attendance session started:", session_key)
    else:
        print("Attendance session cleared")


def update_recognition_count(usn, current_timestamp=None):
    if current_timestamp is None:
        current_timestamp = time.time()

    with attendance_lock:
        previous_time = last_recognition_times.get(usn, 0)

        if (
            current_timestamp - previous_time
            < RECOGNITION_INTERVAL_SECONDS
        ):
            return False

        recognition_counts[usn] += 1
        last_recognition_times[usn] = current_timestamp
        current_count = recognition_counts[usn]

    print(f"Recognition count updated: {usn} -> {current_count}")
    return True


# ================= ATTENDANCE DATABASE WRITES =================

def mark_attendance(
    usn,
    subject_code,
    attendance_date,
    period,
    status,
    entry_time,
    recognition_count,
):
    """
    Inserts or updates exactly one attendance record.
    This function is called only during finalization, never directly
    after a single camera-frame recognition.
    """
    conn = None

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        c.execute("""
            INSERT INTO attendance(
                usn,
                subject_code,
                date,
                period,
                status,
                entry_time,
                recognition_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(usn, subject_code, date, period)
            DO UPDATE SET
                status = excluded.status,
                entry_time = excluded.entry_time,
                recognition_count = excluded.recognition_count
        """, (
            usn,
            subject_code,
            attendance_date,
            period,
            status,
            entry_time,
            int(recognition_count),
        ))

        conn.commit()
        return True

    except sqlite3.Error as error:
        if conn is not None:
            conn.rollback()
        print(f"Attendance write error for {usn}:", error)
        return False

    finally:
        if conn is not None:
            conn.close()


def finalize_attendance(subject_row, session_key, counts_snapshot):
    if not subject_row or not session_key:
        return False

    with attendance_lock:
        if session_key in finalized_sessions:
            return False

    conn = None

    try:
        subject_code = subject_row["subject_code"]
        period = int(subject_row["period"])
        attendance_date = datetime.now().strftime("%Y-%m-%d")
        finalization_time = datetime.now().strftime("%H:%M:%S")

        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT usn
            FROM students
            ORDER BY usn
        """)

        all_students = c.fetchall()
        conn.close()
        conn = None

        successful_writes = 0

        for student_row in all_students:
            usn = student_row[0]
            count = int(counts_snapshot.get(usn, 0))

            if count >= MIN_RECOGNITION_COUNT:
                status = "Present"
                entry_time = finalization_time
            else:
                status = "Absent"
                entry_time = finalization_time

            if mark_attendance(
                usn=usn,
                subject_code=subject_code,
                attendance_date=attendance_date,
                period=period,
                status=status,
                entry_time=entry_time,
                recognition_count=count,
            ):
                successful_writes += 1

        with attendance_lock:
            finalized_sessions.add(session_key)

        print(
            "Attendance finalized:",
            subject_code,
            "Period",
            period,
            f"({successful_writes}/{len(all_students)} records)",
        )

        return True

    except (sqlite3.Error, KeyError, TypeError, ValueError) as error:
        print("Attendance finalization error:", error)
        return False

    finally:
        if conn is not None:
            conn.close()


# ================= FRAME-BASED STUDENT RECOGNITION =================

def recognize_students_from_frame(frame, gray, current_subject, now=None):
    global active_attendance_session

    if now is None:
        now = datetime.now()

    if lbph_recognizer is None or not lbph_label_map:
        cv2.putText(
            frame,
            "Attendance model unavailable",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return frame

    if not is_attendance_window_active(current_subject, now):
        return frame

    session_key = build_attendance_session_key(current_subject, now)

    if active_attendance_session != session_key:
        reset_attendance_session(session_key)

    faces = attendance_face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )

    for x, y, width, height in faces:
        face_roi = gray[y:y + height, x:x + width]

        if face_roi.size == 0:
            continue

        face_roi = cv2.resize(face_roi, (200, 200))
        face_roi = cv2.equalizeHist(face_roi)

        try:
            label_id, confidence = lbph_recognizer.predict(face_roi)
        except cv2.error as error:
            print("LBPH prediction error:", error)
            continue

        student = lbph_label_map.get(int(label_id))

        if (
            student is not None
            and confidence <= LBPH_CONFIDENCE_THRESHOLD
        ):
            usn = student["usn"]
            name = student["name"]

            update_recognition_count(usn)

            with attendance_lock:
                current_count = recognition_counts.get(usn, 0)

            label_text = f"{name} | {usn} | Count: {current_count}"
            box_color = (255, 0, 255)

        else:
            label_text = "Unknown"
            box_color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            box_color,
            2,
        )

        cv2.putText(
            frame,
            label_text,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            box_color,
            2,
        )

    cv2.putText(
        frame,
        "Attendance Verification Active",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 255),
        2,
    )

    return frame


def process_attendance_workflow(frame, gray):
    global last_active_subject
    global last_active_session_key
    global last_active_class_end

    now = datetime.now()
    current_subject = get_current_subject(now)

    if current_subject:
        try:
            session_key = build_attendance_session_key(
                current_subject,
                now,
            )

            _, _, class_end = get_class_times(
                current_subject,
                now,
            )

            if (
                last_active_session_key is not None
                and last_active_session_key != session_key
                and last_active_session_key not in finalized_sessions
            ):
                with attendance_lock:
                    previous_counts = dict(recognition_counts)

                finalize_attendance(
                    last_active_subject,
                    last_active_session_key,
                    previous_counts,
                )

                reset_attendance_session()

            last_active_subject = dict(current_subject)
            last_active_session_key = session_key
            last_active_class_end = class_end

            if is_attendance_window_active(current_subject, now):
                frame = recognize_students_from_frame(
                    frame,
                    gray,
                    current_subject,
                    now,
                )

        except (ValueError, TypeError, KeyError) as error:
            print("Attendance workflow error:", error)

    if (
        last_active_subject is not None
        and last_active_session_key is not None
        and has_class_ended(last_active_class_end, now)
        and last_active_session_key not in finalized_sessions
    ):
        with attendance_lock:
            counts_snapshot = dict(recognition_counts)

        finalize_attendance(
            last_active_subject,
            last_active_session_key,
            counts_snapshot,
        )

        reset_attendance_session()

        last_active_subject = None
        last_active_session_key = None
        last_active_class_end = None

    return frame


# ================= CLASSROOM LOGGING =================

def save_to_db():
    global last_saved_time
    global session_id

    if time.time() - last_saved_time < 30:
        return

    last_saved_time = time.time()
    conn = None

    try:
        now = datetime.now()
        new_session = now.strftime("%Y%m%d_%H")

        if new_session != session_id:
            session_id = new_session

        current_subject = get_current_subject(now)

        if current_subject:
            subject_code = current_subject["subject_code"]
            period = current_subject["period"]
        else:
            subject_code = "No Class"
            period = 0

        students = int(stats["students"])
        attentive = int(stats["attentive"])
        distracted = int(stats["distracted"])

        engagement = (
            round((attentive / students) * 100)
            if students > 0
            else 0
        )

        stats["engagement"] = int(engagement)

        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        c.execute("""
            INSERT INTO logs(
                session,
                date,
                time,
                subject_code,
                period,
                students,
                attentive,
                distracted,
                engagement
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            subject_code,
            period,
            students,
            attentive,
            distracted,
            int(engagement),
        ))

        conn.commit()
        print("Classroom log saved successfully")

    except sqlite3.Error as error:
        if conn is not None:
            conn.rollback()
        print("Classroom-log database error:", error)

    finally:
        if conn is not None:
            conn.close()


# ================= VIDEO STREAM =================

def generate():
    """
    One classroom camera is used for both engagement and attendance.
    Attendance recognition receives the same img/gray frames generated here.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not opening")
        return

    try:
        while True:
            ret, img = cap.read()

            if not ret:
                time.sleep(0.05)
                continue

            img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            results = face_detector.process(rgb)

            students = 0
            attentive = 0
            distracted = 0

            if results.detections:
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    image_height, image_width, _ = img.shape

                    x = int(bbox.xmin * image_width)
                    y = int(bbox.ymin * image_height)
                    box_width = int(bbox.width * image_width)
                    box_height = int(bbox.height * image_height)

                    x = max(0, x)
                    y = max(0, y)
                    box_width = min(box_width, image_width - x)
                    box_height = min(box_height, image_height - y)

                    if box_width <= 0 or box_height <= 0:
                        continue

                    students += 1

                    roi_gray = gray[
                        y:y + box_height,
                        x:x + box_width,
                    ]

                    if roi_gray.size == 0:
                        continue

                    eyes = eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                    )

                    if len(eyes) == 0:
                        if i not in sleep_timers:
                            sleep_timers[i] = time.time()

                        sleep_duration = (
                            time.time() - sleep_timers[i]
                        )

                        if sleep_duration > 3:
                            distracted += 1
                            color = (0, 0, 255)
                            status = "Sleeping"
                        else:
                            attentive += 1
                            color = (0, 255, 255)
                            status = "Blinking"

                    else:
                        sleep_timers[i] = time.time()
                        attentive += 1
                        color = (0, 255, 0)
                        status = "Attentive"

                    cv2.rectangle(
                        img,
                        (x, y),
                        (x + box_width, y + box_height),
                        color,
                        2,
                    )

                    cv2.putText(
                        img,
                        status,
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            engagement = (
                round((attentive / students) * 100)
                if students > 0
                else 0
            )

            global stats

            stats = {
                "students": int(students),
                "attentive": int(attentive),
                "distracted": int(distracted),
                "engagement": int(engagement),
            }

            save_to_db()

            # Attendance uses the exact same classroom frame.
            img = process_attendance_workflow(img, gray)

            cv2.putText(
                img,
                f"Students: {students}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                img,
                f"Engagement: {engagement}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            encoded, buffer = cv2.imencode(".jpg", img)

            if not encoded:
                continue

            frame = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

    finally:
        cap.release()
        print("Classroom camera released")


# ================= ROUTES =================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (
            request.form["username"] == USERNAME
            and request.form["password"] == PASSWORD
        ):
            session["user"] = USERNAME
            return redirect("/home")

    return render_template("login.html")


@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")

    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats_api():
    return jsonify(stats)


@app.route("/graph")
def graph():
    return render_template("graph.html")


@app.route("/sessions")
def sessions_api():
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        df = pd.read_sql_query(
            "SELECT DISTINCT session FROM logs ORDER BY session DESC",
            conn,
        )
        return jsonify(df["session"].tolist())

    finally:
        conn.close()


@app.route("/data")
def data():
    session_req = request.args.get("session")
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        if session_req:
            df = pd.read_sql_query(
                "SELECT * FROM logs WHERE session=?",
                conn,
                params=(session_req,),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM logs ORDER BY ROWID DESC LIMIT 50",
                conn,
            )

        return jsonify({
            "time": df["time"].tolist(),
            "engagement": df["engagement"].astype(int).tolist(),
        })

    finally:
        conn.close()


@app.route("/logs")
def show_logs():
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        c = conn.cursor()
        c.execute("SELECT * FROM logs")
        return str(c.fetchall())

    finally:
        conn.close()


@app.route("/check_logs_structure")
def check_logs_structure():
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        c = conn.cursor()
        c.execute("PRAGMA table_info(logs)")
        return str(c.fetchall())

    finally:
        conn.close()


@app.route("/check_logs")
def check_logs():
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        c = conn.cursor()
        c.execute("SELECT * FROM logs")
        return str(c.fetchall())

    finally:
        conn.close()


@app.route("/subjects")
def subjects():
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        c = conn.cursor()
        c.execute("SELECT * FROM subjects")
        return str(c.fetchall())

    finally:
        conn.close()


@app.route("/report")
def report():
    return render_template("report.html")


# ================= REGISTER STUDENT =================

def _registration_temp_dir(usn):
    """Return the temporary folder used while capturing a student's images."""
    return os.path.join(REGISTRATION_TEMP_FOLDER, str(usn).strip().upper())


def _clear_registration_temp(usn):
    """Delete unfinished temporary images for one student."""
    temp_dir = _registration_temp_dir(usn)

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def _registration_snapshot():
    """Return a safe copy of the current registration state."""
    with registration_lock:
        return {
            "active": bool(registration_state["active"]),
            "usn": registration_state["usn"],
            "name": registration_state["name"],
            "edit_mode": bool(
                registration_state.get("edit_mode", False)
            ),
            "count": int(registration_state["count"]),
            "images": list(registration_state["images"]),
        }


@app.route("/register_student")
def register_student():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "register.html",
        required_images=REGISTRATION_REQUIRED_IMAGES,
        edit_mode=False,
        student=None,
    )


@app.route("/registration/start", methods=["POST"])
def registration_start():
    if "user" not in session:
        return jsonify({
            "success": False,
            "message": "Login required.",
        }), 401

    data = request.get_json(silent=True) or {}

    usn = str(data.get("usn", "")).strip().upper()
    name = str(data.get("name", "")).strip()
    edit_mode = bool(data.get("edit_mode", False))

    if not usn:
        return jsonify({
            "success": False,
            "message": "Please enter the USN.",
        }), 400

    if not name:
        return jsonify({
            "success": False,
            "message": "Please enter the student name.",
        }), 400

    conn = None

    try:
        conn = sqlite3.connect(DATABASE_PATH)

        existing_student = conn.execute(
            "SELECT usn FROM students WHERE usn=?",
            (usn,),
        ).fetchone()

    except sqlite3.Error as error:
        return jsonify({
            "success": False,
            "message": f"Student lookup failed: {error}",
        }), 500

    finally:
        if conn is not None:
            conn.close()

    if edit_mode and existing_student is None:
        return jsonify({
            "success": False,
            "message": "Student record was not found.",
        }), 404

    if not edit_mode and existing_student is not None:
        return jsonify({
            "success": False,
            "message": (
                "This USN is already registered. "
                "Use Edit to replace the student's face images."
            ),
        }), 409

    temp_dir = _registration_temp_dir(usn)
    _clear_registration_temp(usn)
    os.makedirs(temp_dir, exist_ok=True)

    with registration_lock:
        registration_state.update({
            "active": True,
            "usn": usn,
            "name": name,
            "edit_mode": edit_mode,
            "count": 0,
            "images": [],
        })

    return jsonify({
        "success": True,
        "message": (
            "Student update started successfully."
            if edit_mode
            else "Registration started successfully."
        ),
        "required_images": REGISTRATION_REQUIRED_IMAGES,
    })


@app.route("/registration/capture", methods=["POST"])
def registration_capture():
    if "user" not in session:
        return jsonify({
            "success": False,
            "accepted": False,
            "message": "Login required.",
        }), 401

    data = request.get_json(silent=True) or {}

    with registration_lock:
        if not registration_state["active"]:
            return jsonify({
                "success": False,
                "accepted": False,
                "message": "No registration is active.",
            }), 400

        usn = registration_state["usn"]
        current_count = int(registration_state["count"])

    if current_count >= REGISTRATION_REQUIRED_IMAGES:
        return jsonify({
            "success": True,
            "accepted": False,
            "completed": True,
            "count": current_count,
            "message": "All required images are already captured.",
        })

    try:
        frame = _decode_data_url(data.get("image", ""))
        color_photo, training_photo, message = (
            _validate_and_prepare_face(frame)
        )

    except (ValueError, base64.binascii.Error) as error:
        return jsonify({
            "success": False,
            "accepted": False,
            "message": str(error),
            "count": current_count,
        }), 400

    if color_photo is None or training_photo is None:
        return jsonify({
            "success": True,
            "accepted": False,
            "message": message,
            "count": current_count,
            "completed": False,
        })

    with registration_lock:
        if not registration_state["active"]:
            return jsonify({
                "success": False,
                "accepted": False,
                "message": "Registration was cancelled.",
            }), 409

        # Check again because another request could have completed meanwhile.
        if registration_state["count"] >= REGISTRATION_REQUIRED_IMAGES:
            count = int(registration_state["count"])

            return jsonify({
                "success": True,
                "accepted": False,
                "completed": True,
                "count": count,
                "message": "All required images are already captured.",
            })

        registration_state["images"].append({
            "color": color_photo,
            "training": training_photo,
        })
        registration_state["count"] += 1
        count = int(registration_state["count"])

    # Save the normalized grayscale face used by LBPH training.
    temp_dir = _registration_temp_dir(usn)
    os.makedirs(temp_dir, exist_ok=True)

    image_path = os.path.join(temp_dir, f"{count:02d}.jpg")

    try:
        with open(image_path, "wb") as image_file:
            image_file.write(training_photo)

    except OSError as error:
        # Roll back the just-added in-memory image.
        with registration_lock:
            if registration_state["count"] == count:
                registration_state["count"] -= 1

                if registration_state["images"]:
                    registration_state["images"].pop()

        return jsonify({
            "success": False,
            "accepted": False,
            "message": f"Could not save captured image: {error}",
            "count": max(0, count - 1),
        }), 500

    return jsonify({
        "success": True,
        "accepted": True,
        "completed": count >= REGISTRATION_REQUIRED_IMAGES,
        "count": count,
        "message": (
            f"Image {count} of {REGISTRATION_REQUIRED_IMAGES} "
            "captured successfully."
        ),
    })


@app.route("/registration/finish", methods=["POST"])
def registration_finish():
    if "user" not in session:
        return jsonify({
            "success": False,
            "message": "Login required.",
        }), 401

    registration = _registration_snapshot()

    if not registration["active"]:
        return jsonify({
            "success": False,
            "message": "No active registration.",
        }), 400

    usn = registration["usn"]
    name = registration["name"]
    edit_mode = registration["edit_mode"]
    captured_count = registration["count"]
    images = registration["images"]

    if captured_count < REGISTRATION_REQUIRED_IMAGES:
        return jsonify({
            "success": False,
            "message": (
                f"Only {captured_count} of "
                f"{REGISTRATION_REQUIRED_IMAGES} images were captured."
            ),
        }), 400

    if len(images) < REGISTRATION_REQUIRED_IMAGES:
        return jsonify({
            "success": False,
            "message": "Captured image data is incomplete.",
        }), 400

    temp_dir = _registration_temp_dir(usn)
    final_dir = os.path.join(DATASET_DIR, usn)

    image_files = sorted(
        path for path in Path(temp_dir).glob("*.jpg")
        if path.is_file()
    )

    if len(image_files) < REGISTRATION_REQUIRED_IMAGES:
        return jsonify({
            "success": False,
            "message": (
                "The ten temporary training images were not found. "
                "Please cancel and register again."
            ),
        }), 400

    profile_photo = images[0]["color"]
    created_at = datetime.now().isoformat(timespec="seconds")

    conn = None

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("PRAGMA foreign_keys=ON")

        if edit_mode:
            cursor = conn.execute(
                """
                UPDATE students
                SET name=?, photo=?
                WHERE usn=?
                """,
                (name, profile_photo, usn),
            )

            if cursor.rowcount == 0:
                raise sqlite3.IntegrityError(
                    "The student record no longer exists."
                )

            conn.execute(
                "DELETE FROM student_images WHERE usn=?",
                (usn,),
            )

        else:
            conn.execute(
                """
                INSERT INTO students(usn, name, photo)
                VALUES (?, ?, ?)
                """,
                (usn, name, profile_photo),
            )

        for image_number, image_data in enumerate(
            images[:REGISTRATION_REQUIRED_IMAGES],
            start=1,
        ):
            conn.execute(
                """
                INSERT INTO student_images(
                    usn,
                    image_number,
                    photo,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    usn,
                    image_number,
                    image_data["color"],
                    created_at,
                ),
            )

        conn.commit()

    except sqlite3.Error as error:
        if conn is not None:
            conn.rollback()

        return jsonify({
            "success": False,
            "message": f"Database update failed: {error}",
        }), 500

    finally:
        if conn is not None:
            conn.close()

    # Replace the old dataset only after the database transaction succeeds.
    backup_dir = final_dir + "_backup"

    try:
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)

        if os.path.isdir(final_dir):
            os.replace(final_dir, backup_dir)

        os.replace(temp_dir, final_dir)

        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)

    except OSError as error:
        # Try to restore the previous dataset.
        if not os.path.isdir(final_dir) and os.path.isdir(backup_dir):
            os.replace(backup_dir, final_dir)

        return jsonify({
            "success": False,
            "message": (
                "Student data was saved, but the training dataset "
                f"could not be replaced: {error}"
            ),
        }), 500

    trained, training_message = train_all_registered_faces()

    if trained:
        model_loaded = load_lbph_model()
    else:
        model_loaded = False

    _reset_registration_state()

    return jsonify({
        "success": True,
        "message": (
            (
                "Student details and 10 face images were updated. "
                if edit_mode
                else "Student and 10 face images were registered. "
            )
            + training_message
        ),
        "model_loaded": model_loaded,
        "training_success": trained,
    })


@app.route("/registration/cancel", methods=["POST"])
def registration_cancel():
    snapshot = _registration_snapshot()

    if snapshot["usn"]:
        _clear_registration_temp(snapshot["usn"])

    _reset_registration_state()

    return jsonify({
        "success": True,
        "message": "Registration cancelled. No new student data was saved.",
    })


# ================= TIMETABLE =================

@app.route("/view_timetable")
def view_timetable():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()

        c.execute("""
            SELECT *
            FROM timetable
            ORDER BY day, period
        """)

        timetable = c.fetchall()

        return render_template(
            "view_timetable.html",
            timetable=timetable,
        )

    finally:
        conn.close()


@app.route("/add_timetable", methods=["GET", "POST"])
def add_timetable():
    if request.method == "POST":
        conn = None

        try:
            conn = sqlite3.connect(DATABASE_PATH)
            c = conn.cursor()

            c.execute("""
                INSERT INTO timetable(
                    subject_code,
                    day,
                    start_time,
                    end_time,
                    period
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                request.form["subject_code"],
                request.form["day"],
                request.form["start_time"],
                request.form["end_time"],
                request.form["period"],
            ))

            conn.commit()
            return "Timetable Added Successfully"

        except sqlite3.Error as error:
            if conn is not None:
                conn.rollback()
            return f"Timetable database error: {error}", 500

        finally:
            if conn is not None:
                conn.close()

    return render_template("timetable.html")


@app.route("/edit_timetable/<int:id>", methods=["GET", "POST"])
def edit_timetable(id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()

        if request.method == "POST":
            c.execute("""
                UPDATE timetable
                SET
                    subject_code=?,
                    day=?,
                    start_time=?,
                    end_time=?,
                    period=?
                WHERE id=?
            """, (
                request.form["subject_code"],
                request.form["day"],
                request.form["start_time"],
                request.form["end_time"],
                request.form["period"],
                id,
            ))

            conn.commit()
            return redirect("/view_timetable")

        c.execute(
            "SELECT * FROM timetable WHERE id=?",
            (id,),
        )

        row = c.fetchone()

        return render_template(
            "edit_timetable.html",
            row=row,
        )

    except sqlite3.Error as error:
        conn.rollback()
        return f"Timetable database error: {error}", 500

    finally:
        conn.close()


@app.route("/delete_timetable/<int:id>")
def delete_timetable(id):
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        c = conn.cursor()

        c.execute(
            "DELETE FROM timetable WHERE id=?",
            (id,),
        )

        conn.commit()
        return redirect("/view_timetable")

    except sqlite3.Error as error:
        conn.rollback()
        return f"Timetable database error: {error}", 500

    finally:
        conn.close()


# ================= ATTENDANCE DASHBOARD =================

@app.route("/attendance/<subject_code>")
def attendance_dashboard(subject_code):
    if "user" not in session:
        return redirect("/")

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()

        c.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT date, period
                FROM attendance
                WHERE subject_code=?
            )
        """, (subject_code,))

        total_classes = int(c.fetchone()[0] or 0)

        c.execute("""
            SELECT
                s.usn,
                s.name,
                COUNT(
                    CASE
                        WHEN a.status='Present' THEN 1
                    END
                ) AS classes_present
            FROM students AS s
            LEFT JOIN attendance AS a
                ON s.usn=a.usn
               AND a.subject_code=?
            GROUP BY s.usn, s.name
            ORDER BY s.usn
        """, (subject_code,))

        rows = c.fetchall()
        attendance_data = []

        for row in rows:
            present = int(row["classes_present"] or 0)

            percentage = (
                round((present / total_classes) * 100, 2)
                if total_classes > 0
                else 0
            )

            attendance_data.append({
                "name": row["name"],
                "usn": row["usn"],
                "present": present,
                "total": total_classes,
                "percentage": percentage,
            })

        return render_template(
            "attendance.html",
            students=attendance_data,
            subject_code=subject_code,
        )

    finally:
        conn.close()


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ================= STUDENT MANAGEMENT =================

@app.route("/view_students")
def view_students():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()
        c.execute("SELECT * FROM students")

        return render_template(
            "view_students.html",
            students=c.fetchall(),
        )

    finally:
        conn.close()


@app.route("/student_photo/<usn>")
def student_photo(usn):
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        c = conn.cursor()

        c.execute(
            "SELECT photo FROM students WHERE usn=?",
            (usn,),
        )

        row = c.fetchone()

    finally:
        conn.close()

    if row and row[0]:
        return send_file(
            io.BytesIO(row[0]),
            mimetype="image/jpeg",
        )

    return "No Photo", 404



@app.route("/delete_student/<usn>")
def delete_student(usn):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student_images WHERE usn=?", (usn,))
        cursor.execute("DELETE FROM attendance WHERE usn=?", (usn,))
        cursor.execute("DELETE FROM students WHERE usn=?", (usn,))
        conn.commit()

        student_dir = os.path.join(DATASET_DIR, usn)
        legacy_path = os.path.join(DATASET_DIR, f"{usn}.jpg")
        if os.path.isdir(student_dir):
            shutil.rmtree(student_dir, ignore_errors=True)
        if os.path.isfile(legacy_path):
            os.remove(legacy_path)

        train_all_registered_faces()
        return redirect("/view_students")

    except (sqlite3.Error, OSError) as error:
        conn.rollback()
        return f"Student deletion error: {error}", 500

    finally:
        conn.close()


@app.route("/update_student", methods=["POST"])
def update_student():
    data = request.get_json() or {}

    usn = data.get("usn", "").strip()
    name = data.get("name", "").strip()
    image = data.get("image", "")

    if not usn or not name:
        return jsonify({
            "message": "USN and name are required"
        }), 400

    conn = None

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        if image:
            photo = base64.b64decode(
                image.split(",", 1)[1]
            )

            c.execute("""
                UPDATE students
                SET name=?, photo=?
                WHERE usn=?
            """, (
                name,
                photo,
                usn,
            ))

            # The profile photo is updated in SQLite. The 10-image
            # recognition dataset remains unchanged unless the student
            # is registered again.

        else:
            c.execute("""
                UPDATE students
                SET name=?
                WHERE usn=?
            """, (
                name,
                usn,
            ))

        conn.commit()

        return jsonify({
            "message": (
                "Student Updated Successfully. "
                "The existing 10-image recognition dataset is preserved."
            )
        })

    except (
        sqlite3.Error,
        ValueError,
        base64.binascii.Error,
    ) as error:
        if conn is not None:
            conn.rollback()

        return jsonify({
            "message": f"Student update error: {error}"
        }), 500

    finally:
        if conn is not None:
            conn.close()


# ================= ATTENDANCE RECORDS =================

@app.route("/attendance_records")
def attendance_records():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()

        c.execute("""
            SELECT
                attendance.id,
                attendance.usn,
                students.name,
                attendance.subject_code,
                attendance.date,
                attendance.period,
                attendance.status,
                attendance.entry_time,
                attendance.recognition_count
            FROM attendance
            LEFT JOIN students
                ON attendance.usn=students.usn
            ORDER BY
                attendance.date DESC,
                attendance.period DESC,
                attendance.usn ASC
        """)

        return render_template(
            "attendance_records.html",
            records=c.fetchall(),
        )

    finally:
        conn.close()


@app.route("/attendance_recognition_status")
def attendance_recognition_status():
    with attendance_lock:
        counts = dict(recognition_counts)

    return jsonify({
        "active_session": active_attendance_session,
        "minimum_recognition_count": MIN_RECOGNITION_COUNT,
        "recognition_interval_seconds": RECOGNITION_INTERVAL_SECONDS,
        "recognition_counts": counts,
        "model_loaded": lbph_recognizer is not None,
        "label_count": len(lbph_label_map),
    })


# ================= GRAPH DATA =================

@app.route("/period_graph_data")
def period_graph_data():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()

        c.execute("""
            SELECT
                period,
                CAST(ROUND(AVG(engagement), 0) AS INTEGER)
                    AS engagement,
                CAST(ROUND(AVG(distracted), 0) AS INTEGER)
                    AS distracted,
                CAST(ROUND(AVG(attentive), 0) AS INTEGER)
                    AS attentive,
                CAST(ROUND(AVG(students), 0) AS INTEGER)
                    AS students
            FROM logs
            WHERE subject_code != 'No Class'
              AND period IS NOT NULL
              AND period != 0
            GROUP BY period
            ORDER BY period
        """)

        rows = c.fetchall()

        return jsonify({
            "periods": [
                f"Period {row['period']}"
                for row in rows
            ],
            "engagement": [
                int(row["engagement"] or 0)
                for row in rows
            ],
            "distracted": [
                int(row["distracted"] or 0)
                for row in rows
            ],
            "attentive": [
                int(row["attentive"] or 0)
                for row in rows
            ],
            "students": [
                int(row["students"] or 0)
                for row in rows
            ],
        })

    finally:
        conn.close()

@app.route("/edit_student/<string:usn>")
def edit_student(usn):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        student = conn.execute(
            "SELECT usn, name FROM students WHERE usn=?",
            (usn,),
        ).fetchone()

    finally:
        conn.close()

    if student is None:
        return "Student not found", 404

    return render_template(
        "register.html",
        required_images=REGISTRATION_REQUIRED_IMAGES,
        edit_mode=True,
        student=dict(student),
    )

# ================= PERIOD REPORT =================

@app.route("/period_report_data")
def period_report_data():
    """
    Uses rounded averages across all 30-second samples in a class.
    Rounded average is recommended because it represents the whole class,
    unlike the latest value, and is less inflated than the maximum value.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()

        c.execute("""
            SELECT
                date,
                subject_code,
                period,
                CAST(ROUND(AVG(students), 0) AS INTEGER)
                    AS students_present,
                CAST(ROUND(AVG(attentive), 0) AS INTEGER)
                    AS attentive,
                CAST(ROUND(AVG(distracted), 0) AS INTEGER)
                    AS distracted,
                CAST(ROUND(AVG(engagement), 0) AS INTEGER)
                    AS engagement
            FROM logs
            WHERE subject_code != 'No Class'
              AND period IS NOT NULL
              AND period != 0
            GROUP BY date, subject_code, period
            ORDER BY date DESC, period ASC
        """)

        rows = c.fetchall()

        return jsonify([
            {
                "date": row["date"],
                "subject_code": row["subject_code"],
                "period": int(row["period"]),
                "students_present": int(
                    row["students_present"] or 0
                ),
                "attentive": int(
                    row["attentive"] or 0
                ),
                "engagement": int(
                    row["engagement"] or 0
                ),
                "distracted": int(
                    row["distracted"] or 0
                ),
            }
            for row in rows
        ])

    finally:
        conn.close()


# ================= STARTUP =================

init_db()
migrate_attendance_table()
insert_default_subjects()
load_lbph_model()


# ================= RUN =================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
    )
