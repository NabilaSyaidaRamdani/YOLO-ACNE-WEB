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
    <div style='
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    '>
        <h1 style='color: #D63384;'>💖 Welcome to AcneVision</h1>
        <p style='font-size: 18px; color: #555555;'>
            Detect your acne type easily and get personalized skincare tips 🌷<br>
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <img src="https://i.ibb.co/ZTkj9hC/cloud-divider.png" style="width: 100%;"/>
    </div>
""", unsafe_allow_html=True)
""")
st.markdown("""
    <div style='
        background-color: #CDE8FF;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
    '>
        <h2 style='color: #3E80D8;'>💖 Join the AcneVision family</h2>
        <p>Subscribe untuk tips perawatan kulit & info produk terbaru 🌷</p>
    </div>
""", unsafe_allow_html=True)

name = st.text_input("Nama")
email = st.text_input("Email")
if st.button("Subscribe"):
    st.success(f"Terima kasih, {name}! 💌 Kami akan mengirim update ke {email}.")

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

    # Emoji dictionary for each acne type
    emoji_dict = {
        "whitehead": "⚪",
        "blackhead": "⚫",
        "papule": "🔴",
        "nodule": "🟣",
        "pustule": "🟡"
    }

    for label in acne_count:
        icon = emoji_dict.get(label, "🌸")
        st.markdown(f"### {icon} **Detected {label.capitalize()} ({acne_count[label]}x)**")

        if label == "whitehead":
            st.markdown("""
            - ✨ **Komedo Putih Tips**:
              - 🧽 Eksfoliasi rutin (2-3x/minggu) dengan **Salicylic Acid (BHA)**
              - ❄️ Gunakan produk yang mengandung **Benzoyl Peroxide** untuk peradangan
            - 🧴 **Rekomendasi Produk**:
              - *CeraVe Renewing SA Cleanser* — Salicylic Acid, Ceramides, Hyaluronic Acid
            """)
        
        elif label == "blackhead":
            st.markdown("""
            - ✨ **Komedo Hitam Tips**:
              - 🧼 Gunakan cleanser dengan **Salicylic Acid**
              - 🌿 Gunakan toner dengan **Witch Hazel**
            - 🧴 **Rekomendasi Produk**:
              - *The Ordinary Salicylic Acid 2% Solution* — Salicylic Acid, Witch Hazel Extract
            """)
        
        elif label == "papule":
            st.markdown("""
            - ✨ **Papule Tips**:
              - 🚫 Jangan dipencet!
              - 💊 Gunakan **Benzoyl Peroxide** gel/cream
            - 🧴 **Rekomendasi Produk**:
              - *CeraVe Acne Foaming Cream Cleanser* — Benzoyl Peroxide, Niacinamide
            """)
        
        elif label == "nodule":
            st.markdown("""
            - ✨ **Nodul Tips**:
              - 🔬 Konsultasikan ke dokter kulit
              - 💊 Retinoid oral & antibiotik bila diperlukan
            - 🧴 **Rekomendasi Produk**:
              - *Cetaphil PRO Oil Removing Foam Wash* — Zinc Gluconate, Glycerin
            """)
        
        elif label == "pustule":
            st.markdown("""
            - ✨ **Pustule Tips**:
              - ❌ Hindari memencet!
              - 💧 Gunakan produk kombinasi **Benzoyl Peroxide** + **Salicylic Acid**
            - 🧴 **Rekomendasi Produk**:
              - *Neutrogena Clear Pore Cleanser/Mask* — Benzoyl Peroxide (3.5%), Kaolin Clay
            """)

# 🎀 Sidebar input
source = st.sidebar.radio("📷 Pilih Sumber Deteksi:", ["Webcam", "Upload Video", "Upload Gambar"])

# 🌸 Tambahan bunga-bunga cantik dan efek glassmorphism
st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 2px solid rgba(255, 255, 255, 0.3);
    }
    </style>

    <div style='text-align: center; font-size: 20px;'>🌺 🌼 🌸 🌷 🌻 🌹</div>
    <div style='text-align: center; color: #D63384; font-size: 14px;'>
        <em>Stay glowing ✨</em>
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

        # Deteksi menggunakan model
        result_img_bgr, labels = plot_boxes(frame_bgr, model)

        # 👉 Konversi kembali ke RGB untuk ditampilkan di Streamlit
        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

        # 🌸 Buat dua kolom untuk gambar
        col1, col2 = st.columns(2)

        with col1:
            st.image(frame_rgb, caption="Gambar Asli 💁", use_container_width=True)

        with col2:
            st.image(result_img_rgb, caption="Hasil Deteksi Jerawat 💆", use_container_width=True)

        # Tampilkan rekomendasi di bawah gambar
        if labels:
            st.markdown("## 🌟 Rekomendasi Perawatan")
            show_recommendations(labels)

        st.balloons()
