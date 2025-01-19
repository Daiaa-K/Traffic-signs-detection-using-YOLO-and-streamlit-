import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os
from moviepy.editor import VideoFileClip

# Load the YOLOv8 model
model = YOLO(r"results/runs/detect/train/weights/best.pt")

def process_image(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR (if necessary)
    if img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    results = model(img_array)
    
    result_image = results[0].plot()
    
    # Convert BGR back to RGB
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Save the result image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        Image.fromarray(result_image_rgb).save(temp_file.name)
        return temp_file.name
        
def process_video(video_path):
    temp_output_path = tempfile.mktemp(suffix='.mp4')
    final_output_path = tempfile.mktemp(suffix='.mp4')

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        output_frame = results[0].plot()
        out.write(output_frame)
        
        # Update progress bar
        progress_bar.progress((i + 1) / frame_count)
    
    cap.release()
    out.release()
    
    # Convert video to Streamlit-compatible format
    clip = VideoFileClip(temp_output_path)
    clip.write_videofile(final_output_path, codec='libx264')
    clip.close()
    
    os.unlink(temp_output_path)
    
    return final_output_path

st.title("YOLOv8 Object Detection")

upload_type = st.radio("Select input type:", ("Image", "Video"))

if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                result_image = process_image(image)
            st.image(result_image, caption="Detection Result", use_column_width=True)

elif upload_type == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        if st.button("Detect Objects"):
            with st.spinner("Processing video..."):
                progress_bar = st.progress(0)
                output_video_path = process_video(tfile.name)
            st.success("Processing complete!")
            st.video(output_video_path)
            os.unlink(tfile.name)
            os.unlink(output_video_path)
