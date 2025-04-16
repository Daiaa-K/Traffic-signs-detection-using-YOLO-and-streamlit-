# 🚦 Real-Time Traffic Sign Detection with YOLOv8 & Streamlit  

**Detect traffic signs in images, videos, and webcam streams using state-of-the-art object detection**  

![demo](https://github.com/Daiaa-K/Traffic-signs-detection-using-YOLO-and-streamlit-/assets/62758448/ae7d5e0b-2f66-4f4d-8c3e-6f0c3f0a6a7f)  

## 🚀 Features  
- **Multi-input support**:video files (MP4/AVI), and images (JPG/PNG)  
- **Customizable thresholds**: Adjust confidence levels (0.25-1.0)  
- **Auto-download**: YOLOv8n weights downloaded on first run  
- **Device optimization**: Automatic CUDA GPU detection with CPU fallback  

## 📦 Dependencies  
```python
streamlit==1.22.0      # Web interface
ultralytics==8.0.0     # YOLOv8 implementation
opencv-python==4.7.0   # Video/image processing
torch>=1.7.0           # PyTorch backend
