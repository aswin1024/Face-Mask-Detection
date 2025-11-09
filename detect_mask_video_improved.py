# detect_mask_video_improved.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = 'mask_detector_mobilenet_final.h5'
THRESH = 0.6  # probability threshold for "Mask" vs "No Mask"

model = tf.keras.models.load_model(MODEL_PATH)

# Use OpenCV's DNN face detector (more robust than Haar)
proto = cv2.data.haarcascades + "deploy.prototxt"  # placeholder - better to use actual DNN files
# --- fallback to Haar if you don't have DNN model files:
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert and detect faces (Haar fallback)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # consistent preprocessing
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_arr = preprocess_input(face_resized.astype("float32"))
        face_input = np.expand_dims(face_arr, axis=0)

        prob = float(model.predict(face_input)[0][0])  # sigmoid -> prob in [0,1]
        # interpret: closer to 1 -> No Mask (depends on your label mapping). Check class indices:
        # If train_flow.class_indices = {'with_mask':0, 'without_mask':1} then >THRESH means 'No Mask'
        label = ""
        color = (0,255,0)
        # adjust according to your class mapping: print(train_flow.class_indices) to confirm
        if prob >= THRESH:
            label = f"No Mask: {prob:.2f}"
            color = (0,0,255)
        else:
            label = f"Mask: {1-prob:.2f}"
            color = (0,255,0)

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
