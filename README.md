# 🚦 Traffic Sign Detection using YOLOv8

This project provides a user-friendly Streamlit web app for detecting traffic signs in images and videos using a custom fine-tuned YOLOv8 model.

## 🔍 Features

- Upload and analyze **images** and **videos**.
- Detect traffic signs using a fine-tuned YOLOv8 model.
- Visualize detection results directly in the browser.
- Clean, interactive interface built with Streamlit.

## 🧠 Model

The detection is powered by **YOLOv8**, which was **fine-tuned on a dataset of traffic signs** to improve accuracy for this specific use case. The custom-trained weights (`best.pt`) are used to perform detection on user-uploaded media.

## ▶️ How It Works

1. Choose whether you want to upload an **image** or a **video**.
2. Upload your file via the Streamlit interface.
3. Click **"Detect Objects"**.
4. The app will process the file
