import streamlit as st

# 🌸 HARUS DITEMPATKAN DI AWAL
st.set_page_config(page_title="Acne Detection", layout="wide")

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import tempfile
from PIL import Image

# 🌼 CSS Background lucu dan gaya imut
st.markdown("""
    <style>
    body {
        background-image: url("https://i.pinimg.com/originals/1d/b1/2f/1db12f5cb7f421a155b21adf5974c96e.gif");
        background-size: cover;
        background-position: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 0px 20px pink;
    }
    h1, h2, h3 {
        color: #FF69B4;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 🌷 Judul Aplikasi
st.title("🧴 Acne Detection with YOLOv11 💥")
st.markdown("""
    Selamat datang di **deteksi jerawat otomatis**! 😎  
    Yuk cari tahu jenis jerawatmu hanya dengan klik satu tombol 💡  
    Jangan lupa senyum ya! 😊✨
""")

# 💡 Load model YOLOv11
model = YOLO("best.pt")

# 💄 Fungsi untuk menggambar hasil prediksi
def plot_boxes(frame, model):
    results = model.predict(frame, verbose=False)
    annotator = Annotator(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()
            c = int(box.cls[0].item())
            label = model.names[c]

            # Convert to int
            x1, y1, x2, y2 = map(int, b)
            annotator.box_label([x1, y1, x2, y2], f"{label} ✨")

    return annotator.result()

# 🎀 Sidebar input
source = st.sidebar.radio("📷 Pilih Sumber Deteksi:", ["Webcam", "Upload Video", "Upload Gambar"])

# 🖼️ Placeholder untuk output
placeholder = st.empty()

# 🎥 Webcam Mode
if source == "Webcam":
    if st.button("🎬 Mulai Deteksi Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("😥 Tidak bisa membuka kamera.")
        else:
            st.info("Deteksi dimulai... Tampil cakep ya! 😁")
            stop = st.button("🛑 Stop")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop:
                    break
                frame = cv2.resize(frame, (640, 480))
                frame = plot_boxes(frame, model)
                placeholder.image(frame, channels="BGR", use_column_width=True)

            cap.release()
            st.success("✅ Deteksi webcam dihentikan.")

# 🎞️ Upload Video
elif source == "Upload Video":
    uploaded_video = st.file_uploader("📼 Upload video jerawat kamu di sini!", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        st.info("Video diproses... sabar yaa 😎")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            frame = plot_boxes(frame, model)
            placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("🎉 Video selesai diproses!")

# 🖼️ Upload Gambar
elif source == "Upload Gambar":
    uploaded_image = st.file_uploader("🖼️ Upload gambar wajahmu di sini!", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        frame = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        st.image(frame, caption="Gambar Asli 💁", use_column_width=True)

        result_img = plot_boxes(frame, model)
        st.image(result_img, caption="Hasil Deteksi Jerawat 💆", use_column_width=True)
        st.balloons()
