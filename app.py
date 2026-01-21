# --- NEW SCALING FIX ---
    size = (160, 160)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Switch to 0.0 to 1.0 scaling (The most common standard)
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # --- PREDICTION ---
    prediction = model.predict(img_reshape)
    score = float(prediction[0][0])
