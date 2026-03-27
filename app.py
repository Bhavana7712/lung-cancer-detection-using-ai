import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
# model.h5 badulu model.keras ani ivvandi
model = tf.keras.models.load_model("model.h5", compile=False)    

IMG_SIZE = 224

st.title("Lung Cancer Detection Using CNN")

uploaded_file = st.file_uploader(
    "Upload Lung Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    # Prediction
    prediction = model.predict(img)

    normal_acc = prediction[0][0] * 100
    cancer_acc = (1 - prediction[0][0]) * 100

    if cancer_acc > 50:
        st.error("⚠ Cancer Detected")
        st.write(f"Cancer Accuracy: {cancer_acc:.2f}%")

        st.subheader("🩺 Medical Precautions:")
        st.markdown("""
        - Consult a doctor immediately
        - Avoid smoking and polluted areas
        - Drink 2-3 liters of water daily
        - Sleep 7-8 hours daily
        """)

    else:
        st.success("✅ No Cancer Detected")
        st.write(f"Normal Accuracy: {normal_acc:.2f}%")

        st.subheader("💙 Preventive Health Tips:")
        st.markdown("""
        - Avoid smoking
        - Exercise daily
        - Eat healthy food
        - Regular health checkups
        """)