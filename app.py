import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- SETTINGS ---
st.set_page_config(page_title="Gender Predictor", page_icon="👶")
st.title("👶 Baby Gender Predictor")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # This loads your specific model file
    return tf.keras.models.load_model('gender_predictor.h5', compile=False)

model = load_model()

# --- MAIN APP ---
file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # --- IMAGE PREPROCESSING (The "Fix") ---
    size = (160, 160)
    # 1. Resize and crop to the exact center
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 2. Convert to numbers and NORMALIZE (This is usually why it gets stuck)
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    
    # 3. Shape it for the AI (Batch size, Height, Width, Channels)
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # --- PREDICTION ---
    prediction = model.predict(img_reshape)
    # Get the raw number (usually between 0 and 1)
    score = float(prediction[0][0])
    
    st.divider()
    
    # Check the score. If the model is stuck, it will show a score of exactly 0.5 or 0.0.
    if score > 0.5:
        st.header(f"Result: BOY 💙")
        st.write(f"Confidence: {score:.1%}")
    else:
        st.header(f"Result: GIRL 🩷")
        st.write(f"Confidence: {(1-score):.1%}")
        
    st.caption(f"Raw AI Score: {score:.4f}")
