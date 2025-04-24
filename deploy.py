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
        background-color: #FFFFFF;  /* Putih bersih */
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 0px 20px pink;
        color: #333333; /* Warna teks utama abu tua */
    }
    h1, h2, h3 {
        color: #D63384; /* Pink yang lebih gelap */
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);  /* Bayangan ringan biar makin jelas */
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
            cv2.putText(frame, f"{label.capitalize()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            labels.append(label)  # Add the label to the list

    return frame, labels

def show_recommendations(labels):
    # Count the occurrences of each acne type
    acne_count = {label: labels.count(label) for label in set(labels)}
    
    # Show recommendations for each acne type, only once
    for label in acne_count:
        st.write(f"**Detected {label.capitalize()} ({acne_count[label]} instance(s))**:")
        
        if label == "whitehead":
            st.write("🌟 **Komedo Putih:** Eksfoliasi rutin dengan produk berbasis asam salisilat dan gunakan benzoyl peroxide untuk mengurangi peradangan.")
            st.write("🧴 **CeraVe Renewing SA Cleanser**: Salicylic Acid (BHA), Ceramides, Hyaluronic Acid.")
        elif label == "blackhead":
            st.write("🌟 **Komedo Hitam:** Gunakan pembersih berbasis salicylic acid dan toner dengan Witch Hazel untuk mengecilkan pori-pori.")
            st.write("🧴 **The Ordinary Salicylic Acid 2% Solution**: Salicylic Acid, Witch Hazel Extract.")
        elif label == "papule":
            st.write("🌟 **Papule:** Gunakan gel atau krim dengan benzoyl peroxide dan hindari memencet jerawat!")
            st.write("🧴 **CeraVe Acne Foaming Cream Cleanser**: Benzoyl Peroxide, Niacinamide.")
        elif label == "nodule":
            st.write("🌟 **Nodul:** Perawatan dengan retinoid oral atau antibiotik, dan konsultasikan ke dokter kulit jika diperlukan.")
            st.write("🧴 **Cetaphil PRO Oil Removing Foam Wash**: Zinc Gluconate, Glycerin.")
        elif label == "pustule":
            st.write("🌟 **Pustule:** Gunakan produk dengan benzoyl peroxide dan asam salisilat. Hindari memencetnya, dan pertimbangkan untuk berkonsultasi dengan dokter kulit.")
            st.write("🧴 **Neutrogena Clear Pore Cleanser/Mask**: Benzoyl Peroxide (3.5%), Kaolin Clay.")

# 🎀 Sidebar input
source = st.sidebar.radio("📷 Pilih Sumber Deteksi:", ["Webcam", "Upload Video", "Upload Gambar"])

# 🌸 Tambahan bunga-bunga cantik di bawah pilihan
st.sidebar.markdown("""
    <div style='text-align: center; font-size: 20px;'>🌺 🌼 🌸 🌷 🌻 🌹</div>
    <div style='text-align: center; color: #D63384; font-size: 14px;'>
        Ayo pilih dan temukan jenis jerawatmu~ ✨
    </div>
""", unsafe_allow_html=True)

# 🖼️ Placeholder untuk output
placeholder = st.empty()

# 🎥 Webcam Mode
if source == "Webcam":
    if st.button("🎬 Mulai Deteksi Webcam"):
        cap = cv2.VideoCapture(1)
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

        # 👉 Warna asli (RGB)
        frame_rgb = np.array(image.convert("RGB"))

        # 👉 Untuk deteksi (konversi ke BGR karena OpenCV pakai BGR)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Tampilkan gambar asli
        st.image(frame_rgb, caption="Gambar Asli 💁", use_container_width=True)

        # Deteksi menggunakan model
        result_img_bgr, labels = plot_boxes(frame_bgr, model)

        # 👉 Konversi kembali ke RGB untuk ditampilkan di Streamlit
        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

        # Tampilkan hasil deteksi
        st.image(result_img_rgb, caption="Hasil Deteksi Jerawat 💆", use_container_width=True)

        # Tampilkan rekomendasi
        if labels:
            show_recommendations(labels)

        st.balloons()
