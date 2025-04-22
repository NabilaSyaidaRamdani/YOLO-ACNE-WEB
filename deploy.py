import streamlit as st
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np

# Set konfigurasi halaman Streamlit
st.set_page_config(page_title="Acne Detection", layout="wide")

# Judul aplikasi
st.title("🧴 Acne Detection with YOLOv11")

# Load model YOLOv11
model = YOLO("best.pt")  # pastikan file best.pt ada di folder ini

# Fungsi untuk plot bounding boxes di frame
def plot_boxes(frame, model):
    results = model.predict(frame)
    annotator = Annotator(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()  # koordinat box
            c = int(box.cls[0].item())     # class index
            label = model.names[c]
            annotator.box_label(b, label)

    return annotator.result()

# Sidebar untuk memilih sumber video
source = st.sidebar.selectbox("Pilih Sumber Video:", ["Webcam", "Contoh Video"])

# Tombol mulai
start = st.button("Mulai Deteksi")

# Tempat untuk menampilkan video
placeholder = st.empty()

if start:
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("contoh_video.mp4")  # ganti dengan file video jika perlu

    if not cap.isOpened():
        st.error("Tidak bisa membuka kamera/video.")
    else:
        st.info("Tekan tombol 'Stop' di atas untuk menghentikan.")
        stop_button = st.button("Stop")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            frame = cv2.resize(frame, (640, 480))
            frame = plot_boxes(frame, model)

            # Tampilkan ke Streamlit
            placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("Deteksi dihentikan.")

import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera tidak bisa dibuka")
else:
    print("Kamera terbuka")

