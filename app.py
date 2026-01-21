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
# Replace this with the ID of your 'Ultrasound_Data' folder
SHARED_FOLDER_ID = "13kqP6xYq8BJfW9ofuS2eP_if8sWb8GGK" 

# --- GOOGLE DRIVE FUNCTION ---
def save_to_drive(img_bytes, filename):
    try:
        # 1. Setup Credentials
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build('drive', 'v3', credentials=creds)
        
        # 2. Upload the file
        file_metadata = {
            'name': filename, 
            'parents': [SHARED_FOLDER_ID]
        }
        media = MediaIoBaseUpload(io.BytesIO(img_bytes), mimetype='image/png')
        file = service.files().create(
            body=file_metadata, 
            media_body=media, 
            fields='id',
            # This is the "Magic" line to bypass quota issues in some setups
            supportsAllDrives=True 
        ).execute()
        
        # 3. TRANSFER OWNERSHIP (The 403 Fix)
        # Replace 'your-email@gmail.com' with your actual Gmail address
        user_permission = {
            'type': 'user',
            'role': 'owner',
            'emailAddress': 'your-email@gmail.com' 
        }
        service.permissions().create(
            fileId=file.get('id'),
            body=user_permission,
            transferOwnership=True,
            supportsAllDrives=True
        ).execute()
        
        return True
    except Exception as e:
        st.error(f"Error saving to Drive: {e}")
        return False

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('gender_predictor.h5', compile=False)

with st.spinner('Waking up the AI...'):
    model = load_model()

# --- MAIN APP ---
file = st.file_uploader("Upload an ultrasound photo...", type=["jpg", "png", "jpeg"])

if file is not None:
    # Load and show original image
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Original Upload', use_container_width=True)
    
    # --- PREPROCESSING FOR AI ---
    enhancer = ImageEnhance.Contrast(image)
    processed_img = enhancer.enhance(1.5)
    processed_img = ImageOps.grayscale(processed_img).convert('RGB')
    
    size = (160, 160)
    image_resized = ImageOps.fit(processed_img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # --- PREDICTION ---
    prediction = model.predict(img_reshape)
    score = float(prediction[0][0])
    
    # --- DYNAMIC LABELS ---
    if score < 0.5:
        result_text = "BOY"
        result_emoji = "💙"
        confidence = (1 - score)
    else:
        result_text = "GIRL"
        result_emoji = "🩷"
        confidence = score

    st.divider()
    st.header(f"AI Result: {result_text} {result_emoji}")
    st.write(f"Confidence: {confidence:.1%}")
    st.caption(f"Raw AI Score: {score:.4f}")

    # --- PREPARE IMAGE BYTES ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    
    # --- AUTO-SAVE RESULT ---
    auto_filename = f"PREDICTED_{result_text}_{timestamp}.png"
    if save_to_drive(img_bytes, auto_filename):
        st.success("Prediction logged to Drive ✅")

    # --- CORRECTION BUTTONS ---
    st.divider()
    st.subheader("🛠️ Correction & Training")
    st.write("If the AI got it wrong, click the correct button below:")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Actually a BOY 💙"):
            correction_name = f"CORRECTED_BOY_{timestamp}.png"
            if save_to_drive(img_bytes, correction_name):
                st.info("Marked as BOY. Thank you! 🙏")

    with col2:
        if st.button("Actually a GIRL 🩷"):
            correction_name = f"CORRECTED_GIRL_{timestamp}.png"
            if save_to_drive(img_bytes, correction_name):
                st.info("Marked as GIRL. Thank you! 🙏")
