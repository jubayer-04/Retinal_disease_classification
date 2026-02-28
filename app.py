import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import matplotlib.pyplot as plt

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="jubayer009/retinal_efficientnetv2b3.keras",
        filename="retina_efficientnetv2b3.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.title("Retinal Disease Detection")

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    if st.button("Predict"):
        prediction = model.predict(img_array)
        probabilitites = prediction[0]
        class_index = np.argmax(probabilitites)
        predicted_class = class_names[class_index]
        confidence = np.max(probabilitites)
    
        st.subheader("predicted Result")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: {confidence * 100:.2f}%")

        st.subheader("Confidence for All Classes")

        fig = plt.figure()
        plt.bar(class_names, probabilities * 100)
        plt.xlabel("Classes")
        plt.ylabel("Confidence (%)")
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
    
        st.pyplot(fig)










