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

# --- GOOGLE DRIVE SETUP ---
# Make sure your 'google-credentials.json' is still in your GitHub folder!
SHARED_FOLDER_ID = "YOUR_FOLDER_ID_HERE" # Put your folder ID back here

def save_to_drive(img_bytes, filename):
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': filename, 'parents': [SHARED_FOLDER_ID]}
        media = MediaIoBaseUpload(io.BytesIO(img_bytes), mimetype='image/png')
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return True
    except Exception as e:
        st.error(f"Error saving to Drive: {e}")
        return False

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('gender_predictor.h5', compile=False)

model = load_model()

# --- MAIN APP ---
file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('RGB')
    
    # --- ENHANCED PREPROCESSING ---
    enhancer = ImageEnhance.Contrast(image)
    image_processed = enhancer.enhance(1.5)
    image_processed = ImageOps.grayscale(image_processed).convert('RGB')
    
    st.image(image, caption='Uploaded Ultrasound', use_container_width=True)
    
    # Resize & Normalize
    size = (160, 160)
    image_resized = ImageOps.fit(image_processed, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # --- PREDICTION ---
    prediction = model.predict(img_reshape)
    score = float(prediction[0][0])
    
    # --- LOGIC ---
    # Based on our last test (where 0.44 was a BOY), we use this mapping:
    if score < 0.5:
        result_text = "BOY"
        result_emoji = "💙"
        confidence = (1 - score)
    else:
        result_text = "GIRL"
        result_emoji = "🩷"
        confidence = score

    st.divider()
    st.header(f"Result: {result_text} {result_emoji}")
    st.write(f"Confidence: {confidence:.1%}")
    st.caption(f"Raw AI Score: {score:.4f}")

    # --- SAVE TO DRIVE ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_{result_text}_{confidence:.0%}.png"
    
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    
    if save_to_drive(buf.getvalue(), file_name):
        st.success("Result saved to Google Drive! ✅")
