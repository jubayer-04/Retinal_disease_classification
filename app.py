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
with st.sidebar.expander("Model Details"):
    st.write("Input Shape:", model.input_shape)
    st.write("Output Shape:", model.output_shape)
    st.write("Total Parameters:", f"{model.count_params():,}")
    st.write("Classes:", class_names)
    
st.write("All classes are: ", class_names)

st.markdown("Sample Fundus Retinal Images")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(Image.open("samples/cataract.jpg"), caption="Cataract", width=150)

with col2:
    st.image(Image.open("samples/diabetic_retinopathy.jpeg"), caption="Diabetic Retinopathy", width=150)

with col3:
    st.image(Image.open("samples/glaucoma.jpg"), caption="Glaucoma", width=150)

with col4:
    st.image(Image.open("samples/normal.jpg"), caption="Normal", width=150)


uploaded_file = st.file_uploader("Upload Fundus Retinal Image", type=["jpg", "png", "jpeg"])


if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    left, center, right = st.columns([1,2,1])

    with center:
        st.image(image, caption="Uploaded Image")

    # Create 3 small columns inside center to center button
        b1, b2, b3 = st.columns([1,2,1])
        with b2:
            predict_clicked = st.button("Predict")
        
    
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
   
    if predict_clicked:
    # your prediction code here
        prediction = model.predict(img_array)
        probabilitites = prediction[0]
        class_index = np.argmax(probabilitites)
        predicted_class = class_names[class_index]
        confidence = np.max(probabilitites)
        
        st.subheader("predicted Result")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: {confidence * 100:.2f}%")

        if predicted_class == 'cataract':
            st.text("Suggestion: Possible cataract detected. Please consult an ophthalmologist for a detailed eye examination, early treatment or surgery can restore vision effectively.")
        elif predicted_class == "diabetic_retinopathy":
            st.text("Suggestion: Indicators of diabetic retinopathy found. Maintain blood sugar control and consult an eye doctor for retinal evaluation and timely treatment.")
        elif predicted_class == "glaucoma":
            st.text("Suggestion: Signs of glaucoma detected. Visit an eye specialist as soon as possible for pressure testing and treatment to prevent permanent vision loss.")
        else:
            st.text("Suggestion: No major abnormalities detected. Continue regular eye checkups and maintain healthy eye care habits.")

        st.subheader("Disclaimer:")
        st.markdown("**This result is AI-assisted and not a medical diagnosis. Please consult a qualified doctor for confirmation. Remember Ai can make mistakes...! Don't trust it blindly......**")
            
    
        st.subheader("Confidence for All Classes")
    
        fig = plt.figure(figsize=(3.5, 2.2))  # smaller figure

        plt.bar(class_names, probabilitites * 100, width=1)
        
        plt.xticks(rotation=45, fontsize=5)
        plt.yticks(fontsize=5)
        
        plt.xlabel("Classes", fontsize=5)
        plt.ylabel("Confidence (%)", fontsize=5)
        
        plt.ylim(0, 100)
        
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=False)

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





st.subheader("Model Performance")
st.text("Overall Accuracy: 98.5%")
st.text("Overall Precision: 0.9856")
st.text("Overall Recall: 0.985")
st.text("Overall F1 Score: 0.985")
st.text(report_text)

st.subheader("Model Description")
st.text("We have worked with EfficientNetV2B3 model which is a convolutional neural network architecture that employs fused MBConv blocks and compound scaling to optimize accuracy–efficiency trade-offs while reducing training time. It leverages progressive learning and depth–width–resolution scaling to improve feature representation with fewer parameters. In this work, the model is fine-tuned via transfer learning on retinal fundus images for robust multiclass disease classification.")






































































