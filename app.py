import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Gender Predictor", page_icon="👶")
st.title("👶 Ultrasound Gender Predictor")
st.write("Powered by MobileNetV2 Transfer Learning")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('gender_predictor.h5')

model = load_my_model()

uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', width=300)
    
    st.write("Analyzing...")

    # NEW PROCESSING FOR MOBILENET (RGB, 160x160)
    if image_display.mode != "RGB":
        image_display = image_display.convert("RGB")
        
    img = image_display.resize((160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    st.divider()
    if confidence > 0.5:
        st.success(f"### Prediction: MALE 💙")
        st.progress(int(confidence * 100))
        st.write(f"**Confidence:** {confidence:.1%}")
    else:
        st.success(f"### Prediction: FEMALE 🩷")
        st.progress(int((1-confidence) * 100))
        st.write(f"**Confidence:** {(1-confidence):.1%}")
