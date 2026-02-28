import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import matplotlib.pyplot as plt


class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']


col1, col2 = st.columns([1,1.2])

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
with st.sidebar.expander("Model Details"):
    st.write("Input Shape:", model.input_shape)
    st.write("Output Shape:", model.output_shape)
    st.write("Total Parameters:", f"{model.count_params():,}")
    st.write("Classes:", class_names)

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])
st.write("All classes are: ", class_names)

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
    
        fig = plt.figure(figsize=(5, 3))
        plt.bar(class_names, probabilitites * 100)
        plt.xlabel("Classes")
        plt.ylabel("Confidence (%)")
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        st.pyplot(fig)

report_text = f"""
{'Class':<40}{'Precision':<10}{"|  "}{'Recall':<10}{"|  "}{'F1-Score':<10}{"|  "}{'Support':<10}
{'-'*70}
{'cataract':<40}{0.99 :<10.2f}{"|  "}{0.98:<10.2f}{"|  "}{0.98:<10.2f}{"|  "}{100:<10}
{'-'*70}
{'diabetic_retinopathy':<25}{1.00:<10.2f}{"|  "}{1.00:<10.2f}{"|  "}{1.00:<10.2f}{"|  "}{100:<10}
{'-'*70}
{'glaucoma':<40}{0.95:<10.2f}{"|  "}{1.00:<10.2f}{"|  "}{0.98:<10.2f}{"|  "}{100:<10}
{'-'*70}
{'normal':<40}{1.00:<10.2f}{"|  "}{0.96:<10.2f}{"|  "}{0.98:<10.2f}{"|  "}{100:<10}
"""



st.text("Accuracy: 98.5%")
st.text("Precision: 0.9856")
st.text("Recall: 0.985")
st.text("F1 Score: 0.985")

st.subheader("Model Performance (Test Set)")
st.text(report_text)








































