import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

# 🌸 HARUS DITEMPATKAN DI AWAL
st.set_page_config(page_title="Acne Detection", layout="wide")

# 🎨 Gradient Background
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #FFFAF0 0%, #E6F7FF 100%);
}
</style>
""", unsafe_allow_html=True)

# 🌼 Welcome Banner with Frosted Glass
st.markdown("""
<div style='
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 24px rgba(0,0,0,0.1);
'>
    <h1 style='color: #D63384;'>💖 Welcome to AcneVision</h1>
    <p style='font-size: 18px; color: #555555;'>
        Detect your acne type easily and get personalized skincare tips 🌷
    </p>
</div>
""", unsafe_allow_html=True)

# 🌟 Five Vertical Feature Boxes (Pastel + Shadow)
st.markdown("""
<div style='display: flex; flex-direction: column; gap: 1rem; margin-top: 2rem;'>
  <div style='background: #FFEBEE; padding:1rem; border-radius:12px; box-shadow:0 8px 16px rgba(255,235,238,0.5); text-align:center; font-weight:bold; color:#C62828;'>⚡ Fast</div>
  <div style='background: #F3E5F5; padding:1rem; border-radius:12px; box-shadow:0 8px 16px rgba(243,229,245,0.5); text-align:center; font-weight:bold; color:#6A1B9A;'>🔬 Accurate</div>
  <div style='background: #E8F5E9; padding:1rem; border-radius:12px; box-shadow:0 8px 16px rgba(232,245,233,0.5); text-align:center; font-weight:bold; color:#2E7D32;'>🌷 Personalized</div>
  <div style='background: #E1F5FE; padding:1rem; border-radius:12px; box-shadow:0 8px 16px rgba(225,245,254,0.5); text-align:center; font-weight:bold; color:#0277BD;'>🔒 Safe</div>
  <div style='background: #FFF3E0; padding:1rem; border-radius:12px; box-shadow:0 8px 16px rgba(255,243,224,0.5); text-align:center; font-weight:bold; color:#EF6C00;'>🎉 Fun</div>
</div>
""", unsafe_allow_html=True)

# 💡 Load YOLOv11 model
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
            cv2.putText(frame, f"{label.capitalize()}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)
            labels.append(label)
    return frame, labels

# Recommendations function
def show_recommendations(labels):
    acne_count = {label: labels.count(label) for label in set(labels)}
    emoji_dict = {"whitehead":"⚪","blackhead":"⚫","papule":"🔴","nodule":"🟣","pustule":"🟡"}
    for label, count in acne_count.items():
        icon = emoji_dict.get(label)
        st.markdown(f"### {icon} **{label.capitalize()} ({count})**")
        # Detailed tips...

# 🎀 Sidebar for uploads & tips
source = st.sidebar.radio("📷 Pilih Sumber Deteksi:", ["Upload Video","Upload Gambar"])

st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] { background:rgba(255,255,255,0.7); backdrop-filter:blur(8px);} 
</style>
<div style='text-align:center; font-size:20px;'>🌺 🌸 🌼</div>
<br>
""", unsafe_allow_html=True)

uploaded_video=None; uploaded_image=None
with st.sidebar:
    if source=="Upload Video":
        if st.button("🎥 Pilih Video"): uploaded_video=st.file_uploader("Upload video...",type=["mp4","avi"])
    else:
        if st.button("🖼️ Pilih Gambar"): uploaded_image=st.file_uploader("Upload gambar...",type=["jpg","png"])
    # Sidebar tips
    st.markdown("""
    <div style='background:#fff0f5;padding:16px;border-radius:12px;box-shadow:0 8px 16px rgba(255,235,238,0.3);margin-top:2rem;'>
      <h4 style='color:#D63384;'>💡 Skincare Tips</h4>
      🌞 Pagi: Cleanser + Sunscreen<br>
      🌙 Malam: Double cleansing + Serum
    </div>
    """, unsafe_allow_html=True)

# Placeholder
placeholder = st.empty()

# Video processing
if source=="Upload Video" and uploaded_video:
    tfile=tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap=cv2.VideoCapture(tfile.name)
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        img, labels = plot_boxes(frame, model)
        placeholder.image(img, channels="BGR", use_container_width=True)
    cap.release()

# Image processing
elif source=="Upload Gambar" and uploaded_image:
    img = Image.open(uploaded_image)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_out, labels = plot_boxes(frame, model)
    rgb_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2)
    with col1: st.image(img, caption="Asli",use_container_width=True)
    with col2: st.image(rgb_out, caption="Hasil",use_container_width=True)

# 📬 Contact Footer
st.markdown("""
<div style='text-align:center; margin-top:3rem;'>
  <a href='https://github.com/username' target='_blank' style='margin:0 1rem; font-size:24px;'>🐙</a>
  <a href='https://instagram.com/username' target='_blank' style='margin:0 1rem; font-size:24px; color:#E4405F;'>📸</a>
  <a href='mailto:email@example.com' style='margin:0 1rem; font-size:24px; color:#333;'>✉️</a>
</div>
""", unsafe_allow_html=True)
