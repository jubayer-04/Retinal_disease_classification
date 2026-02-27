import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Retinal Disease CLassification", layout='centered')
st.title("Retinal Image Classification")

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
model = load_model('retina_efficientnetv2b3.keras')

uploaded_file = st.file_uploader("Upload a Retinal Fundus Image", type=['jpg','png','jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224,224))
    st.image(image, caption='Uploaded Image', width=500)
    
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis = 0)


    if st.button("Predict"):
    #Prediction
        prediction = model.predict(image)
        probabilitites = prediction[0]
        class_index = np.argmax(probabilitites)
        predicted_class = class_names[class_index]
        confidence = np.max(probabilitites)

        st.subheader("predicted Result")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence: .2f}**")

        st.subheader("Class Confidence Levels")

        fig, ax = plt.subplots()
        ax.bar(class_names, probabilitites)
        ax.set_ylabel("Confidence")
        ax.set_ylim([0,1])
        plt.xticks(rotation=45)

        st.pyplot(fig)

