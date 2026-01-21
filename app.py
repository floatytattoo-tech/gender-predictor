import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageEnhance
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
    
    # --- ENHANCED PREPROCESSING ---
    # 1. Boost Contrast & Sharpness (Help the AI see edges)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    st.image(image, caption='AI View (Enhanced)', use_container_width=True)
    
    # 2. Resize
    size = (160, 160)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 3. Normalize (Trying the 0.0 to 1.0 range)
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # --- PREDICTION ---
    prediction = model.predict(img_reshape)
    score = float(prediction[0][0])
    
    st.divider()

    # Let's use a "Standard" mapping for now:
    # Most models: 0 = Boy, 1 = Girl OR 0 = Girl, 1 = Boy.
    # If your score is 0.44 and it's a Boy, then 0.0 -> 0.5 IS Boy.
    if score < 0.5:
        st.header(f"Result: BOY 💙")
        st.write(f"Confidence: {(1-score):.1%}")
    else:
        st.header(f"Result: GIRL 🩷")
        st.write(f"Confidence: {score:.1%}")
        
    st.caption(f"Raw AI Score: {score:.4f}")
