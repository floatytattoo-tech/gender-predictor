import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Gender Predictor", page_icon="👶")
st.title("👶 Baby Gender Predictor")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('gender_predictor.h5', compile=False)

model = load_model()

file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # --- PRECISION PREPROCESSING ---
    size = (160, 160)
    # Using a high-quality resize method
    image_resized = ImageOps.fit(image, size, Image.Resampling.BICUBIC)
    
    # Convert to array and use MobileNet-style scaling (-1 to 1)
    # Some models prefer this over (0 to 1)
    img_array = np.asarray(image_resized).astype('float32')
    img_array = (img_array / 127.5) - 1.0  
    
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # --- PREDICTION ---
    prediction = model.predict(img_reshape)
    score = float(prediction[0][0])
    
    st.divider()
    
    # LOGIC SWAP CHECK: 
    # If it's still always showing Girl, we may need to flip the logic.
    if score > 0.5:
        st.header(f"Result: BOY 💙")
        st.progress(score)
    else:
        st.header(f"Result: GIRL 🩷")
        st.progress(1.0 - score)
        
    st.write(f"**AI Confidence Score:** {score:.4f}")
    st.caption("Note: If score is near 0.0 or 1.0 every time, the image might need better cropping.")
