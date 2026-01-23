import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import io
import time  # Added this for the timestamp

# Google Drive Imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# --- SETTINGS ---
st.set_page_config(page_title="Gender Predictor", page_icon="👶")
# --- CONFIGURATION ---
# Replace the text inside the quotes with your real Folder ID
SHARED_FOLDER_ID = SHARED_FOLDER_ID = "1UU6_GMp9SX9i5Z5GMyp99-e_lpJWzop4"

# --- GOOGLE DRIVE FUNCTION ---
def save_to_drive(img_bytes, filename):
    try:
        # Check if secrets exist
        if "gcp_service_account" not in st.secrets:
            st.warning("⚠️ Google Drive keys not found in Secrets. Image NOT saved.")
            return False

        # Authenticate with Google
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build('drive', 'v3', credentials=creds)

        # Upload file
        file_metadata = {'name': filename, 'parents': [SHARED_FOLDER_ID]}
        media = MediaIoBaseUpload(io.BytesIO(img_bytes), mimetype='image/png')
        service.files().create(body=file_metadata, media_body=media).execute()
        return True

    except Exception as e:
        st.error(f"⚠️ Drive Sync Error: {e}")
        return False
        
st.title("👶 Baby Gender Predictor")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('gender_predictor.h5', compile=False)
    

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

    # 7. Correction Buttons
    st.divider()
    st.subheader("🛠️ Correction & Training")
    st.write("Help train the AI by clicking the correct gender if it was wrong:")

    # --- SAVE TO DRIVE LOGIC (Automatic) ---
        # We need to define timestamp here so the buttons below can use it!
        timestamp = int(time.time())
        save_filename = f"PREDICTION_{res_text}_{timestamp}.png"

        # This tries to save the prediction automatically
        if save_to_drive(img_data, save_filename):
            st.success("✅ Image saved to Training Data!")

        # --- HEADERS FOR THE BUTTONS ---
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
