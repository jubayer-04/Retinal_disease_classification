import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tflite_runtime.interpreter import Interpreter
import gdown
import os

st.set_page_config(page_title="Retinal Disease Classification", layout='centered')
st.title("Retinal Image Classification")

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# -----------------------------
# Google Drive Download
# -----------------------------
MODEL_PATH = "retina_efficientnetv2b3.tflite"
FILE_ID = "PASTE_YOUR_FILE_ID_HERE"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# Load TFLite Model
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a Retinal Fundus Image",
    type=['jpg','png','jpeg']
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    st.image(image, caption='Uploaded Image', width=500)

    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    if st.button("Predict"):

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        probabilities = prediction[0]
        class_index = np.argmax(probabilities)
        predicted_class = class_names[class_index]
        confidence = float(np.max(probabilities))

        st.subheader("Predicted Result")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        st.subheader("Class Confidence Levels")

        fig, ax = plt.subplots()
        ax.bar(class_names, probabilities)
        ax.set_ylabel("Confidence")
        ax.set_ylim([0,1])
        plt.xticks(rotation=45)

        st.pyplot(fig)
