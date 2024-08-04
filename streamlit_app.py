import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

# Load the YOLOv8 model
model = YOLO(r"results\runs\detect\train\weights\best.pt")

def process_image(image):
    results = model(image)
    return results[0].plot()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        output = results[0].plot()
        stframe.image(output, channels="BGR")
    cap.release()

st.title("Traffic signs Detection")

upload_type = st.radio("Select input type:", ("Image", "Video"))

if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Objects"):
            result_image = process_image(image)
            st.image(result_image, caption="Detection Result", use_column_width=True)

elif upload_type == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        if st.button("Detect Objects"):
            process_video(tfile.name)
