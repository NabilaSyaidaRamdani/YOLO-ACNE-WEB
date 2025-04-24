import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 💡 Load YOLOv11 model
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

# 🌷 Judul Aplikasi
st.title("🧴 Acne Detection with YOLOv11 💥")

# 🎀 Sidebar input
source = st.sidebar.radio("📷 Pilih Sumber Deteksi:", ["Webcam", "Upload Video", "Upload Gambar"])

# 🖼️ Placeholder untuk output
placeholder = st.empty()

# 🎥 Webcam Mode
if source == "Webcam":
    if st.button("🎬 Mulai Deteksi Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("😥 Tidak bisa membuka kamera. Cek izin akses kamera di browser atau pastikan tidak ada aplikasi lain yang menggunakan kamera.")
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
                for label in labels:
                    if label == "Acne Scars":
                        st.write("🌟 **Bekas Jerawat:** Gunakan produk yang mengandung retinoid, pertimbangkan terapi laser, dan jangan lupa selalu menggunakan tabir surya! 😊")
                    elif label == "Enlarged Pores":
                        st.write("🌟 **Pori-pori Membesar:** Gunakan produk dengan niacinamide, lakukan eksfoliasi dengan AHA atau asam salisilat. Jaga kebersihan kulit dengan pembersih lembut!")
                    elif label == "Whitehead":
                        st.write("🌟 **Komedo Putih:** Eksfoliasi rutin dengan produk berbasis asam salisilat dan gunakan benzoyl peroxide untuk mengurangi peradangan.")
                    elif label == "Blackhead":
                        st.write("🌟 **Komedo Hitam:** Gunakan pembersih berbasis salicylic acid dan toner dengan Witch Hazel untuk mengecilkan pori-pori.")
                    elif label == "Papules":
                        st.write("🌟 **Papula:** Gunakan gel atau krim dengan benzoyl peroxide dan hindari memencet jerawat!")
                    elif label == "Nodules":
                        st.write("🌟 **Nodul:** Perawatan dengan retinoid oral atau antibiotik, dan konsultasikan ke dokter kulit jika diperlukan.")
                
            cap.release()
            st.success("✅ Deteksi webcam dihentikan.")

# (Code for Upload Video and Upload Image is unchanged)
