# Face-Mask-Detection# Face Mask Detection using MobileNetV2 🩺😷

This project detects whether a person is wearing a face mask or not in real-time using a webcam.

---

## 🚀 Features
- Real-time mask detection using OpenCV
- Trained on MobileNetV2 (transfer learning)
- Works without external DNN `.prototxt` files
- Uses Haar Cascade for face detection

---

## 🧠 Model
The model is trained using the `MobileNetV2` architecture on a custom dataset of:
- **with_mask/**
- **without_mask/**

---

## 🖥️ Run Instructions

# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Train the model 
python train_mask_detector_mobilenet.py

# 3️⃣ Run real-time detection
python detect_mask_video_improved.py

Press q to quit.

---

## 🧑‍💻 Author
Ashwindev Anoop
