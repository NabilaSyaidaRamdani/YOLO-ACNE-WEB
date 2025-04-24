import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

# 🌸 HARUS DITEMPATKAN DI AWAL
st.set_page_config(page_title="Acne Detection", layout="wide")

# 🌼 CSS Background lucu dan gaya imut dengan warna ocean blue
st.markdown("""
    <style>
    body {
        background-color: #1E90FF;  /* Ocean Blue */
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

def plot_boxes(frame, model):
    results = model.predict(frame, verbose=False)
    labels = []  # List to store detected labels

    for result in results:
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()  # Make sure this is a list or array
            c = int(box.cls[0].item())  # Class index
            label = model.names[c]  # Get the class name from the model

            # Convert the coordinates to integers, and unpack them
            x1, y1, x2, y2 = map(int, b)  # Ensure these are integers

            # Draw the bounding box and label using OpenCV
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw box with red color
            cv2.putText(frame, f"{label} ✨", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            labels.append(label)  # Add the label to the list

    return frame, labels

def show_recommendations(labels):
    for label in labels:
        if label == "Acne Scars":
            st.write("🌟 **Bekas Jerawat:** Gunakan produk yang mengandung retinoid, pertimbangkan terapi laser, dan jangan lupa selalu menggunakan tabir surya! 😊")
        elif label == "Whitehead":
            st.write("🌟 **Komedo Putih:** Eksfoliasi rutin dengan produk berbasis asam salisilat dan gunakan benzoyl peroxide untuk mengurangi peradangan.")
        elif label == "Blackhead":
            st.write("🌟 **Komedo Hitam:** Gunakan pembersih berbasis salicylic acid dan toner dengan Witch Hazel untuk mengecilkan pori-pori.")
        elif label == "Papule":
            st.write("🌟 **Papule:** Gunakan gel atau krim dengan benzoyl peroxide dan hindari memencet jerawat!")
        elif label == "Nodule":
            st.write("🌟 **Nodul:** Perawatan dengan retinoid oral atau antibiotik, dan konsultasikan ke dokter kulit jika diperlukan.")

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
                frame = cv2.resize(frame, (640, 640))
                result_img, labels = plot_boxes(frame, model)
                placeholder.image(result_img, channels="BGR", use_container_width=True)
                
            # Display recommendations
            if labels:
                show_recommendations(labels)

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
            result_img, labels = plot_boxes(frame, model)
            placeholder.image(result_img, channels="BGR", use_container_width=True)
        
        # Display recommendations
        if labels:
            show_recommendations(labels)

        cap.release()
        st.success("🎉 Video selesai diproses!")

# 🖼️ Upload Gambar
elif source == "Upload Gambar":
    uploaded_image = st.file_uploader("🖼️ Upload gambar wajahmu di sini!", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        frame = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize uploaded image to smaller size (e.g., 320x320)
        frame = cv2.resize(frame, (320, 320))

        st.image(frame, caption="Gambar Asli 💁", use_container_width=True)

        # Process and resize the result image to a smaller size
        result_img, labels = plot_boxes(frame, model)
        result_img = cv2.resize(result_img, (320, 320))  # Resize result

        st.image(result_img, caption="Hasil Deteksi Jerawat 💆", use_container_width=True)

        # Display recommendations
        if labels:
            show_recommendations(labels)

        st.balloons()

        st.balloons()
