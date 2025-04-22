from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import streamlit as st
import cv2
from vidgear.gears import CamGear

#Membuat Streamlit UI
st.set_page_config(page_title="ACNE DETECTION", layout="wide", initial_sidebar_state="expanded")

st.title("Acne Detection With yolov11")

#Load Model Yolov11
model = ('best.pt')

#Membuat Function untuk plot bounding boxes pada frames

def plot_boxes(frame, model):
    results = model.predict(frame)
    annotasi = Annotator(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]   # Box Coordinat
            c = box.cls
            annotasi.box_label(b, model.names[int(c)])
        return annotasi.results()

#Membuat Function untuk proces dan display video

def process_video(data, model, placeholder):
    if data == 'Webcam':
        camera = cv2.VideoCapture(0)  #Kode Webcam
    else: 
        camera = CamGear(source=data, stream_mode=True, logging=True).start()
    while True:
        if data == 'Webcam':
            ret, frame = camera.read()
            if not ret:
                break
        else:
            frame = camera.read()
            if frame is None:
                break
            

