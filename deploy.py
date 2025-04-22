import streamlit as st
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import tempfile
from PIL import Image

# Set konfigurasi halaman Streamlit
st.set_page_config(page_title="Acne Detection", layout="wide")

# Judul aplikasi
st.title("🧴 Acne Detection with YOLOv11 💥")
st.markdown("Selamat datang di deteksi jerawat otomatis! 😎 Yuk cari tahu jenis jerawatmu hanya dengan klik satu tombol 💡")

# Load model YOLOv11
model = YOLO("best.pt")  # Pastikan file best.pt ada di folder yang sama

# Fungsi untuk plot bounding boxes di frame
def plot_boxes(frame, model):
    results = model.predict(frame)
    annotator = Annotator(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()  # Koordinat kotak
            c = int(box.cls[0].item())     # Kelas
            label = model.names[c]
            annotator.box_label(b, label + " 🧼")

    return annotator.result()

# Sidebar untuk memilih sumber input
source = st.sidebar.radio("📷 Pilih Sumber Deteksi:", ["Webcam", "Upload Video", "Upload Gambar"])

# Tempat untuk menampilkan output
placeholder = st.empty()

# Webcam Mode
if source == "Webcam":
    start = st.button("🎬 Mulai Deteksi Webcam")
    if start:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("😥 Tidak bisa membuka kamera.")
        else:
            stop_button = st.button("🛑 Stop")
            st.info("Deteksi dimulai... Tampil cakep ya! 😁")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_button:
                    break

                frame = cv2.resize(frame, (640, 480))
                frame = plot_boxes(frame, model)

                placeholder.image(frame, channels="BGR", use_column_width=True)

            cap.release()
            st.success("✅ Deteksi webcam dihentikan.")

# Upload Video Mode
elif source == "Upload Video":
    uploaded_video = st.file_uploader("📼 Upload video jerawat kamu di sini!", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        st.info("Video diproses, sabar ya 😎")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame = plot_boxes(frame, model)
            placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("🎉 Video selesai diproses!")

# Upload Gambar Mode
elif source == "Upload Gambar":
    uploaded_image = st.file_uploader("🖼️ Upload gambar wajahmu di sini!", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        st.image(frame, caption="Gambar Asli", use_column_width=True)

        result_img = plot_boxes(frame, model)
        st.image(result_img, caption="Hasil Deteksi Jerawat 💆", use_column_width=True)
        st.balloons()
