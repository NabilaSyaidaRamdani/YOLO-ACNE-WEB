import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

# 🌸 HARUS DITEMPATKAN DI AWAL
st.set_page_config(page_title="Acne Detection", layout="wide")

# 🎨 CSS: Gradient Background + Banner Frosted Glass
st.markdown("""
    <style>
    /* Entire App Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #FFDEE9 0%, #B5FFFC 100%);
    }
    </style>
""", unsafe_allow_html=True)

# 🌼 Welcome Banner with Frosted Glass Effect
st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); padding: 2rem; border-radius: 20px; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.1);'>
        <h1 style='color: #D63384;'>💖 Welcome to AcneVision</h1>
        <p style='font-size: 18px; color: #555555;'>Detect your acne type easily and get personalized skincare tips 🌷</p>
    </div>
""", unsafe_allow_html=True)

# 🌟 Insert 5 Vertical Feature Boxes with Pastel-Colored Shadows
st.markdown("""
    <div style="display: flex; flex-direction: column; row-gap: 1rem; margin-top: 2rem;">
        <div style="background: #FFCDD2; padding: 1rem; border-radius: 12px; box-shadow: 0 8px 16px rgba(255,205,210,0.4); text-align: center; color: #B71C1C; font-weight: bold;">⚡ Fast</div>
        <div style="background: #F8BBD0; padding: 1rem; border-radius: 12px; box-shadow: 0 8px 16px rgba(248,187,208,0.4); text-align: center; color: #880E4F; font-weight: bold;">🔬 Accurate</div>
        <div style="background: #E1BEE7; padding: 1rem; border-radius: 12px; box-shadow: 0 8px 16px rgba(225,190,231,0.4); text-align: center; color: #4A148C; font-weight: bold;">🌷 Personalized</div>
        <div style="background: #D1C4E9; padding: 1rem; border-radius: 12px; box-shadow: 0 8px 16px rgba(209,196,233,0.4); text-align: center; color: #311B92; font-weight: bold;">🔒 Safe</div>
        <div style="background: #C5CAE9; padding: 1rem; border-radius: 12px; box-shadow: 0 8px 16px rgba(197,202,233,0.4); text-align: center; color: #1A237E; font-weight: bold;">🎉 Fun</div>
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
            cv2.putText(frame, f"{label.capitalize()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            labels.append(label)
    return frame, labels

def show_recommendations(labels):
    acne_count = {label: labels.count(label) for label in set(labels)}
    emoji_dict = {"whitehead": "⚪", "blackhead": "⚫", "papule": "🔴", "nodule": "🟣", "pustule": "🟡"}
    for label, count in acne_count.items():
        icon = emoji_dict.get(label, "🌸")
        st.markdown(f"### {icon} **Detected {label.capitalize()} ({count}x)**")
        if label == "whitehead": st.markdown("""- ✨ **Komedo Putih Tips**: ...""", unsafe_allow_html=True)
        elif label == "blackhead": st.markdown("""- ✨ **Komedo Hitam Tips**: ...""", unsafe_allow_html=True)
        elif label == "papule": st.markdown("""- ✨ **Papule Tips**: ...""", unsafe_allow_html=True)
        elif label == "nodule": st.markdown("""- ✨ **Nodul Tips**: ...""", unsafe_allow_html=True)
        elif label == "pustule": st.markdown("""- ✨ **Pustule Tips**: ...""", unsafe_allow_html=True)

# 🎀 Sidebar input: pilihan & uploader
source = st.sidebar.radio("📷 Pilih Sumber Deteksi:", ["Upload Video", "Upload Gambar"])
st.sidebar.markdown("""
    <style>[data-testid="stSidebar"]{background:rgba(255,255,255,0.6);backdrop-filter:blur(10px);border-right:2px solid rgba(255,255,255,0.3);}</style>
    <div style='text-align:center;font-size:20px;'>🌺 🌼 🌸 🌷 🌻 🌹</div>
    <div style='text-align:center;color:#D63384;font-size:14px;'><em>Stay glowing ✨</em></div><br><br>
""", unsafe_allow_html=True)
uploaded_video=None; uploaded_image=None
with st.sidebar:
    if source=="Upload Video" and st.button("🎥 Klik untuk memilih video"): uploaded_video=st.file_uploader("📼 Upload video jerawat kamu di sini!", type=["mp4","avi","mov"])
    if source=="Upload Gambar" and st.button("🖼️ Klik untuk memilih gambar"): uploaded_image=st.file_uploader("🖼️ Upload gambar wajahmu di sini!", type=["jpg","jpeg","png"])
    st.markdown("""
        <div style='background-color:#fff0f5;padding:16px;border-radius:12px;box-shadow:0 10px 15px rgba(255,182,193,0.4),0 20px 30px rgba(255,182,193,0.3);margin-top:50px;'>
            <h4 style='color:#d63384;'>💡 Skincare Tips & Trick</h4>
            <div style='display:flex;justify-content:space-between;'>
                <div style='width:48%;'><strong>🌞 Pagi</strong><br>🧼 Gentle cleanser<br>☀️ Sunscreen SPF 30+<br>💧 Moisturizer ringan</div>
                <div style='width:48%;'><strong>🌙 Malam</strong><br>🌿 Double cleansing<br>🎯 Serum (AHA-BHA)<br>💤 Night cream</div>
            </div><hr style='margin:10px 0;'>
            <div style='text-align:center;font-size:13px;color:#D63384;'>💕 Rutin itu kunci kulit sehat 💕</div>
        </div>
    """, unsafe_allow_html=True)

# 🖼️ Placeholder untuk hasil
placeholder = st.empty()

# 🎞️ Proses Upload Video
if source=="Upload Video" and uploaded_video:
    tfile=tempfile.NamedTemporaryFile(delete=False); tfile.write(uploaded_video.read()); cap=cv2.VideoCapture(tfile.name)
    st.info("Video diproses... sabar yaa 😎")
    while cap.isOpened(): ret,frame=cap.read();
        if not ret: break
        frame=cv2.resize(frame,(640,480)); img,labels=plot_boxes(frame,model)
        placeholder.image(img,channels="BGR",use_container_width=True)
    if labels: show_recommendations(labels)
    cap.release(); st.success("🎉 Video selesai diproses!")

# 🖼️ Proses Upload Gambar
elif source=="Upload Gambar" and uploaded_image:
    img=Image.open(uploaded_image); frame_rgb=np.array(img.convert("RGB"))
    frame_bgr=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR); img_out,labels=plot_boxes(frame_bgr,model)
    result_rgb=cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB)
    col1,col2=st.columns(2)
    with col1: st.image(frame_rgb,caption="Gambar Asli 💁",use_container_width=True)
    with col2: st.image(result_rgb,caption="Hasil Deteksi 💆",use_container_width=True)
    if labels: st.markdown("## 🌟 Rekomendasi Perawatan"); show_recommendations(labels)
    st.balloons()

# 📬 Contact Section
st.markdown("""
<div style='text-align:center; margin-top:3rem;'>
    <a href='https://github.com/username' target='_blank' style='margin:0 1rem; font-size:24px; color:#333;'>🐙 GitHub</a>
    <a href='https://instagram.com/username' target='_blank' style='margin:0 1rem; font-size:24px; color:#E4405F;'>📸 Instagram</a>
    <a href='mailto:email@example.com' style='margin:0 1rem; font-size:24px; color:#333;'>✉️ Email</a>
</div>
""", unsafe_allow_html=True)
