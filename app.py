import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import datetime

# --- SETTINGS ---
st.set_page_config(page_title="Gender Predictor", page_icon="👶")
st.title("👶 Baby Gender Predictor")

# --- CONFIGURATION ---
# Replace this with your Google Drive folder ID
SHARED_FOLDER_ID = "YOUR_FOLDER_ID_HERE" 

# --- GOOGLE DRIVE FUNCTION ---
def save_to_drive(img_bytes, filename):
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': filename, 'parents': [SHARED_FOLDER_ID]}
        media = MediaIoBaseUpload(io.BytesIO(img_bytes), mimetype='image/png')
        service.files().create(body=file_metadata, media_body=media).execute()
        return True
    except Exception as e:
        # If it still shows 403, Google just needs more time to process your billing
        st.error(f"Drive Sync Status: {e}")
        return False

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('gender_predictor.h5', compile=False)

with st.spinner('Waking up the AI...'):
model = tf.keras.models.load_model('gender_predictor_v3.h5', compile=False)

# --- MAIN APP ---
file = st.file_uploader("Upload an ultrasound photo...", type=["jpg", "png", "jpeg"])

if file is not None:
    # 1. Load Original Image
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Ultrasound', use_container_width=True)
   # --- NEW VALIDATION SECTION ---
    width, height = image.size
    aspect_ratio = width / height
    
    if aspect_ratio > 1.2 or aspect_ratio < 0.8:
        st.warning("⚠️ Image is not square! Please crop to a square around the baby's bottom for better accuracy.")
    
    if width < 200:
        st.error("🚫 Image resolution is too low.")
        st.stop() 
    # ------------------------------
    
    # 2. AI Preprocessing (Contrast + Grayscale)
    enhancer = ImageEnhance.Contrast(image)
    proc_img = enhancer.enhance(1.5)
    proc_img = ImageOps.grayscale(proc_img).convert('RGB')
    
    # 3. Resize & Predict
    size = (160, 160)
    image_resized = ImageOps.fit(proc_img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_reshape)
    score = float(prediction[0][0])
    
    # 4. Result Logic (Calibrated: < 0.5 is BOY)
    if score < 0.5:
        res_text, res_emoji, conf = "BOY", "💙", (1 - score)
    else:
        res_text, res_emoji, conf = "GIRL", "🩷", score

    st.divider()
    st.header(f"AI Result: {res_text} {res_emoji}")
    st.write(f"Confidence: {conf:.1%}")
    st.caption(f"Internal AI Score: {score:.4f}")

    # 5. Prepare for Saving
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_data = buf.getvalue()
    
    # 6. Automatic Log
    if save_to_drive(img_data, f"PREDICTED_{res_text}_{timestamp}.png"):
        st.success("Log saved to Google Drive! ✅")

    # 7. Correction Buttons
    st.divider()
    st.subheader("🛠️ Correction & Training")
    st.write("Help train the AI by clicking the correct gender if it was wrong:")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Actually a BOY 💙"):
            if save_to_drive(img_data, f"CORRECTED_BOY_{timestamp}.png"):
                st.info("Stored as BOY for training. Thanks! 🙏")
    with c2:
        if st.button("Actually a GIRL 🩷"):
            if save_to_drive(img_data, f"CORRECTED_GIRL_{timestamp}.png"):
                st.info("Stored as GIRL for training. Thanks! 🙏")
