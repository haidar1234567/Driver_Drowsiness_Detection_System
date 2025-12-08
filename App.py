# app.py
import streamlit as st
import os
import cv2
import numpy as np
import mediapipe as mp
import pygame
import tensorflow as tf
import tempfile
import time
import re
from datetime import datetime
from twilio.rest import Client
from threading import Lock

# -----------------------------
# Twilio configuration 
# -----------------------------
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

client = Client(TWILIO_SID, TWILIO_AUTH)

# -----------------------------
# Alert sound 
# -----------------------------
WAV_FILE = "alert_sound.wav"
try:
    pygame.mixer.init()
except Exception:
    pygame.mixer.quit()
    pygame.mixer.init()
alert_sound = pygame.mixer.Sound(WAV_FILE)
_sound_lock = Lock()
_sound_active = False

def start_alert_sound():
    global _sound_active
    with _sound_lock:
        if not _sound_active:
            _sound_active = True
            alert_sound.play(loops=-1)

def stop_alert_sound():
    global _sound_active
    with _sound_lock:
        if _sound_active:
            _sound_active = False
            alert_sound.stop()

# -----------------------------
# MediaPipe + model
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.7,
                                  min_tracking_confidence=0.7)

YAWN_THRESHOLD = 0.48
MIN_FRAMES = 5
IMG_SIZE = 86
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
NOSE_TIP = 1
EAR_THRESHOLD = 0.20
CNN_THRESHOLD = 0.35
HEAD_TILT_THRESHOLD = 0.23

# -----------------------------
# Model load 
# -----------------------------
try:
    model = tf.keras.models.load_model("best_eye_model_v2.h5", compile=False)
except Exception:
    model = None  # silently skip CNN if missing

# -----------------------------
# User-friendly markdown
# -----------------------------
st.markdown("<h1 style='text-align: center; color: #00b4d8;'>🚗 Driver Drowsiness & Accident Alert System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Real-time monitoring using AI | Stay Awake, Stay Safe 😴➡️🚨</p>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Developed by Haidar Raza</p>", unsafe_allow_html=True)

# -----------------------------
# Helper functions 
# -----------------------------
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0.0

def get_eye_status(frame, landmarks, eye_indices, w, h):
    x_min = int(min([landmarks[i].x for i in eye_indices]) * w)
    y_min = int(min([landmarks[i].y for i in eye_indices]) * h)
    x_max = int(max([landmarks[i].x for i in eye_indices]) * w)
    y_max = int(max([landmarks[i].y for i in eye_indices]) * h)
    pad = 10
    x_min = max(0, x_min - pad); y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad); y_max = min(h, y_max + pad)
    eye_crop = frame[y_min:y_max, x_min:x_max]
    if eye_crop.size == 0 or model is None:
        return "UNKNOWN", 0.0
    gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.resize(gray_eye, (IMG_SIZE, IMG_SIZE)) / 255.0
    gray_eye = gray_eye.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict(gray_eye, verbose=0)[0][0]
    status = "OPEN" if pred > CNN_THRESHOLD else "CLOSED"
    return status, float(pred)

def calculate_mouth_open_ratio(landmarks):
    try:
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_corner = landmarks[78]
        right_corner = landmarks[308]
        vertical_dist = abs(upper_lip.y - lower_lip.y)
        horizontal_dist = abs(left_corner.x - right_corner.x)
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
    except Exception:
        return 0

# -----------------------------
# Twilio helpers 
# -----------------------------
def verify_number(number, verified_list):
    if not number:
        return ""
    if not re.match(r"^\+\d{10,15}$", number):
        return "Invalid Number"
    return "Number is verified" if number in verified_list else "Number is not verified"

def send_twilio_alert(to_number, message, verified_list):
    if to_number not in verified_list:
        return False
    try:
        client.messages.create(body=message, from_=TWILIO_NUMBER, to=to_number)
        return True
    except Exception:
        return False

# -----------------------------
# Streamlit UI 
# -----------------------------
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = None
if "running" not in st.session_state:
    st.session_state.running = False

st.sidebar.header("Modes / Controls")
if st.sidebar.button("Webcam"):
    st.session_state.selected_mode = "webcam"
if st.sidebar.button("IP Camera"):
    st.session_state.selected_mode = "ipcam"

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Mode:**")
if st.session_state.selected_mode:
    st.sidebar.markdown(f"<span style='color:green;font-weight:bold'>{st.session_state.selected_mode}</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("None")

# -----------------------------
# IP Webcam credentials input
# -----------------------------
ipcam_url = ""
if st.session_state.selected_mode == "ipcam":
    st.sidebar.subheader("IP Webcam Credentials")
    ip = st.sidebar.text_input("IP Address (e.g., 192.168.1.5)")
    port = st.sidebar.text_input("Port (default 8080)", value="8080")
    path = st.sidebar.text_input("Path (default video)", value="video")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if ip and username and password:
        ipcam_url = f"http://{username}:{password}@{ip}:{port}/{path}"
    else:
        st.sidebar.warning("Enter IP, Username, and Password to connect IP Camera.")

tw_number1 = st.sidebar.text_input("Enter Number 1 (with +countrycode...)")
tw_number2 = st.sidebar.text_input("Enter Number 2 (with +countrycode...)")
verified_text = st.sidebar.text_area("Enter verified numbers (one per line)", height=100)
verified_list = [n.strip() for n in verified_text.splitlines() if n.strip()]

if st.sidebar.button("Enable Number (send confirmation)"):
    msgs = []
    for n in (tw_number1, tw_number2):
        if not n:
            msgs.append("No number provided.")
            continue
        status = verify_number(n, verified_list)
        if status != "Number is verified":
            msgs.append(f"{n}: {status}")
            continue
        ok = send_twilio_alert(n, "✅ This number is enabled to receive alert messages.", verified_list)
        msgs.append(f"{n}: {'Sent' if ok else 'Failed'}")
    for m in msgs:
        st.sidebar.info(m)

if st.sidebar.button("Start Detection"):
    st.session_state.running = True
if st.sidebar.button("Stop Detection"):
    st.session_state.running = False
    stop_alert_sound()
    st.sidebar.info("Stopped.")

video_ph = st.empty()
status_ph = st.empty()
yawn_ph = st.empty()
alert_ph = st.empty()

for key, val in {
    "yawn_count": 0, "consecutive": 0, "last_yawn_state": False,
    "heavily_drowsy_count": 0, "heavily_drowsy_start": None,
    "alert_sent": False, "warning_sent": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -----------------------------
# PROCESS FRAME (heavily drowsy + alerts)
# -----------------------------
def process_frame_local(frame):
    mouth_ratio = 0.0
    is_yawn = False
    heavily_drowsy_detected = False
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mouth_ratio = calculate_mouth_open_ratio(face_landmarks.landmark)
                if mouth_ratio > YAWN_THRESHOLD:
                    st.session_state.consecutive += 1
                    if st.session_state.consecutive >= MIN_FRAMES:
                        is_yawn = True
                else:
                    st.session_state.consecutive = 0

                if is_yawn and not st.session_state.last_yawn_state:
                    st.session_state.yawn_count += 1
                    st.session_state.last_yawn_state = True
                elif not is_yawn:
                    st.session_state.last_yawn_state = False

                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
                avg_ear = (left_ear + right_ear)/2.0

                left_status, _ = get_eye_status(frame, face_landmarks.landmark, LEFT_EYE, frame.shape[1], frame.shape[0])
                right_status, _ = get_eye_status(frame, face_landmarks.landmark, RIGHT_EYE, frame.shape[1], frame.shape[0])

                left_eye_y = np.mean([face_landmarks.landmark[i].y for i in LEFT_EYE])
                right_eye_y = np.mean([face_landmarks.landmark[i].y for i in RIGHT_EYE])
                nose_y = face_landmarks.landmark[NOSE_TIP].y
                face_tilt = nose_y - ((left_eye_y + right_eye_y)/2.0)

                eyes_closed = (left_status == "CLOSED" and right_status == "CLOSED") or (avg_ear < EAR_THRESHOLD)
                head_down = face_tilt > HEAD_TILT_THRESHOLD
                heavily_drowsy_detected = eyes_closed or head_down

                # -----------------------------
                # Accident alert logic (4s or 5 times heavily drowsy)
                # -----------------------------
                if heavily_drowsy_detected:
                    if st.session_state.heavily_drowsy_start is None:
                        st.session_state.heavily_drowsy_start = time.time()
                    st.session_state.heavily_drowsy_count += 1
                    elapsed = time.time() - st.session_state.heavily_drowsy_start
                    if ((elapsed >= 4 or st.session_state.heavily_drowsy_count >= 5) 
                        and not st.session_state.alert_sent):
                        for n in verified_list:
                            send_twilio_alert(n, "⚠️ Accident Alert! Driver is heavily drowsy.", verified_list)
                        st.session_state.alert_sent = True
                else:
                    st.session_state.heavily_drowsy_start = None

                # -----------------------------
                # Yawn alert logic (10 yawns)
                # -----------------------------
                if st.session_state.yawn_count >= 10 and not st.session_state.warning_sent:
                    for n in verified_list:
                        send_twilio_alert(n, "⚠️ Warning! Driver is showing frequent yawns.", verified_list)
                    st.session_state.warning_sent = True

                if heavily_drowsy_detected:
                    start_alert_sound()
                else:
                    stop_alert_sound()
                break
    except Exception:
        pass

    try:
        if heavily_drowsy_detected:
            color = (0,0,255); label = "Heavily Drowsy"
        elif is_yawn:
            color = (0,255,255); label = "Lightly Drowsy"
        else:
            color = (0,255,0); label = "Active"
        cv2.putText(frame, f"Status: {label}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Yawn Count: {st.session_state.yawn_count}", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    except Exception:
        pass

    return frame, is_yawn, heavily_drowsy_detected

# -----------------------------
# STREAMING LOOP (UPGRADED IP CAM)
# -----------------------------
if st.session_state.running:
    cap = None
    if st.session_state.selected_mode == "webcam":
        cap = cv2.VideoCapture(0)
    elif st.session_state.selected_mode == "ipcam":
        if ipcam_url:
            cap = cv2.VideoCapture(ipcam_url)
        else:
            st.warning("Enter IP Camera credentials in sidebar.")
            st.session_state.running = False
            st.stop()

    if cap is None or not cap.isOpened():
        st.warning("Camera not accessible. Check IP/username/password.")
    else:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame, _, _ = process_frame_local(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_ph.image(frame_rgb, channels="RGB")
        cap.release()
        stop_alert_sound()
