import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# ------------------------------
# Load Model (cached)
# ------------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="jubayer009/retinal_efficientnetv2b3",
        filename="retina_efficientnetv2b3.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ------------------------------
# UI
# ------------------------------
st.title("Retinal Disease Detection")

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    if st.button("Predict"):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
    
        st.write("Prediction:", predicted_class)
