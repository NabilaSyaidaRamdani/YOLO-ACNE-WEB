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
            Detect your acne type easily and get personalized skincare tips 🌷
        </p>
    </div>
""", unsafe_allow_html=True)

# 💡 Load model YOLOv11
model = YOLO("best.pt")

def plot_boxes(frame, model):
    results = model.predict(frame, verbose=False)
    labels = []
    for result in results:
        for box in result.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = int(box.cls[0].item())
            label = model.names[c]
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label.capitalize()}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            labels.append(label)
    return frame, labels

def show_recommendations(labels):
    acne_count = {label: labels.count(label) for label in set(labels)}
    emoji_dict = {"whitehead":"⚪","blackhead":"⚫","papule":"🔴","nodule":"🟣","pustule":"🟡"}
    for label, count in acne_count.items():
        icon = emoji_dict.get(label, "🌸")
        st.markdown(f"### {icon} *Detected {label.capitalize()} ({count}x)*")
        # detailed tips omitted for brevity

# 🎀 Sidebar input
tt = st.sidebar  # alias
source = tt.radio("📷 Pilih Sumber Deteksi:", ["Upload Video", "Upload Gambar"])

# 🌸 Sidebar styling and tips
tt.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(255,255,255,0.3);
    }
    </style>
    <div style='text-align:center; font-size:20px;'>🌺 🌼 🌸 🌷 🌻 🌹</div>
    <div style='text-align:center; color:#D63384; font-size:14px;'><em>Stay glowing ✨</em></div>
    <br><br>
""", unsafe_allow_html=True)

with tt:
    # push sidebar tips down
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color:#fff0f5; padding:16px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
            <h4 style="color:#d63384;">💡 Skincare Tips & Trick</h4>
            <div style="display:flex; justify-content:space-between;">
                <div style="width:48%;"><strong>🌞 Pagi</strong><br>🧼 Gentle cleanser<br>☀ Sunscreen SPF 30+<br>💧 Moisturizer ringan</div>
                <div style="width:48%;"><strong>🌙 Malam</strong><br>🌿 Double cleansing<br>🎯 Serum (AHA-BHA)<br>💤 Night cream</div>
            </div>
            <hr style="margin:10px 0;">
            <div style='text-align:center; font-size:13px; color:#D63384;'>💕 Rutin itu kunci kulit sehat 💕</div>
        </div>
    """, unsafe_allow_html=True)

# 🖼 Placeholder untuk output
placeholder = st.empty()

# 🎞 Upload Video
if source == "Upload Video":
    # push uploader down
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("📼 Upload video jerawat kamu di sini!", type=["mp4","avi","mov"])
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
        if labels:
            show_recommendations(labels)
        cap.release()
        st.success("🎉 Video selesai diproses!")

# 🖼 Upload Gambar
elif source == "Upload Gambar":
    # push uploader down
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("🖼 Upload gambar wajahmu di sini!", type=["jpg","jpeg","png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        frame_rgb = np.array(image.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        result_img_bgr, labels = plot_boxes(frame_bgr, model)
        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.image(frame_rgb, caption="Gambar Asli 💁", use_container_width=True)
        with col2:
            st.image(result_img_rgb, caption="Hasil Deteksi Jerawat 💆", use_container_width=True)
        if labels:
            st.markdown("## 🌟 Rekomendasi Perawatan")
            show_recommendations(labels)
        st.balloons()
