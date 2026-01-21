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
    return tf.keras.models.load_model('gender_predictor.h5', compile=False)

model = load_model()

# --- MAIN APP ---
file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # --- IMAGE PREPROCESSING ---
    size = (160, 160)
    # This line must be indented exactly 4 spaces (1 tab) under the "if"
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Normalize pixels to 0.0 - 1.0
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # --- PREDICTION ---
    prediction = model.predict(img_reshape)
    score = float(prediction[0][0])
    
    st.divider()
    
    # Logic: 0.5 is the middle ground
    # --- SWAPPED LABELS ---
    if score > 0.5:
        st.header(f"Result: GIRL 🩷")
        st.write(f"Confidence: {score:.1%}")
    else:
        # Since your score was 0.32, it will now fall here and show BOY
        st.header(f"Result: BOY 💙")
        st.write(f"Confidence: {(1-score):.1%}")
        
    st.caption(f"Raw AI Score: {score:.4f}")
