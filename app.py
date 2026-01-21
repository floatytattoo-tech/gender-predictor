import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import datetime

# --- SETTINGS ---
st.title("👶 Baby Gender Predictor")
st.write("Upload an ultrasound image, and the AI will predict the gender!")

# --- CONFIGURATION ---
# PASTE YOUR FOLDER ID BELOW (Keep the quotes!)
PARENT_FOLDER_ID = "13kqP6xYq8BJfW9ofuS2eP_if8sWb8GGK"

# --- GOOGLE DRIVE FUNCTION ---
def upload_to_drive(file_obj, filename):
    try:
        # Load credentials from Streamlit Secrets
        gcp_creds = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(gcp_creds)
        service = build('drive', 'v3', credentials=creds)
        
        # Prepare file metadata with the specific parent folder
        file_metadata = {
            'name': filename,
            'parents': [PARENT_FOLDER_ID]
        }
            
        fh = io.BytesIO()
        file_obj.save(fh, format='PNG')
        fh.seek(0)
        media = MediaIoBaseUpload(fh, mimetype='image/png')
        
        # Upload
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return True
    except Exception as e:
        st.error(f"Error saving to Drive: {e}")
        return False

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('gender_predictor.h5')

with st.spinner('Loading Model...'):
    model = load_model()

# --- MAIN APP ---
file = st.file_uploader("Choose an ultrasound photo...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Predict
    size = (160, 160)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_reshape)
    confidence = prediction[0][0]
    
    if confidence > 0.5:
        st.header(f"It's a BOY! 💙 ({confidence:.0%})")
        label = "BOY"
    else:
        st.header(f"It's a GIRL! 🩷 ({(1-confidence):.0%})")
        label = "GIRL"

    # SAVE TO DRIVE
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{timestamp}_{label}.png"
    
    if upload_to_drive(image, save_name):
        st.success("Image saved to database! ✅")
