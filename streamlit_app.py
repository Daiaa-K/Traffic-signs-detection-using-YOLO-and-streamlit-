import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os
from moviepy.editor import VideoFileClips

# Load the YOLOv8 model
model = YOLO(r"results/runs/detect/train/weights/best.pt")

def process_image(image):
    results = model(image)
    return plot_results(results[0], image)

def plot_results(result, img):
    img = np.array(img)
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = round(box.conf[0].item(), 2)
        cls = result.names[box.cls[0].item()]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'{cls} {conf}'
        t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    return img

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a temporary file for the output video
    output_path = tempfile.mktemp(suffix='.mp4')
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        output_frame = plot_results(results[0], frame)
        out.write(output_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    # Convert the video to a format that Streamlit can display
    clip = VideoFileClip(output_path)
    final_output_path = tempfile.mktemp(suffix='.mp4')
    clip.write_videofile(final_output_path, codec='libx264')
    clip.close()
    
    # Remove the temporary file
    os.unlink(output_path)
    
    return final_output_path

st.title("YOLOv8 Object Detection")

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
            st.text("Processing video... This may take a while.")
            output_video_path = process_video(tfile.name)
            st.text("Processing complete. Displaying result.")
            st.video(output_video_path)
            # Clean up temporary files
            os.unlink(tfile.name)
            os.unlink(output_video_path)
