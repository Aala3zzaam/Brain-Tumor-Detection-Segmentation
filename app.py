import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import gdown


def download_models():
    if not os.path.exists("models"):
        os.makedirs("models")

    detection_path = "models/Brain_Tumor_Detection.h5"
    segmentation_path = "models/Brain_Tumor_Segmentation.h5"

    if not os.path.exists(detection_path):
        gdown.download("https://drive.google.com/file/d/1uTkfDtLkTnNJ-YxsipPfaGhPR2UwQcgv/view?usp=drive_link", detection_path, quiet=False)

    if not os.path.exists(segmentation_path):
        gdown.download("https://drive.google.com/file/d/1bacioDshEhLfSKFHPyDjw79pQBYKPOcK/view?usp=drive_link", segmentation_path, quiet=False)

download_models()


st.set_page_config(layout="wide")
st.markdown("""
    <style>
    div[data-testid="stApp"] {
        background-color: #9ab8d9;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title('Brain Tumor Model')
page = st.sidebar.selectbox("Choose a page", ('Home', 'Tumor Detection', 'Tumor Segmentation'))


if page == 'Home':
       
    image = Image.open("brain.png")  
    resized = image.resize((300, 300))

    col1, col2 = st.columns([1, 2]) 

    with col1:
        st.image(resized)

    with col2:
        st.markdown("<h1 style='margin-top: 50px;'>Brain Tumor Detection and Segmentation</h1>", unsafe_allow_html=True)

    st.markdown("""
    ###  About Brain Tumors

    Brain tumors are abnormal growths of cells in the brain that can be life-threatening if not detected early. MRI scans are commonly used to visualize these tumors.

    ### 🤖 Project Overview

    In this project, the used **pretrained deep learning models**:
    - 🟦 **ResNet** for tumor classification (detecting if there's a tumor or not)
    - 🟥 **U-Net** for tumor segmentation (highlighting tumor regions in MRI scans)

    Using AI, the system can analyze **brain MRI scans** and predict whether a tumor is present, helping in faster and more accurate diagnosis.
    """)
    image2 = Image.open('image.png')
    
    st.image(image2.resize((800,300)))
    image3 = Image.open('image2.jpg')
    st.image(image3.resize((800,300)))



elif page == 'Tumor Detection':
    st.title('🟦 Tumor Detection')
    st.write('Upload a brain MRI scan to check if a tumor is detected.')

    uploaded_file = st.file_uploader('Upload MRI Image', type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image.resize((300, 300)), caption="Uploaded MRI")

        with st.spinner('Processing...'):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            model = tf.keras.models.load_model('models/Brain_Tumor_Detection.h5')
            prediction = model.predict(img_array)[0][0]

            result = "Tumor Detected" if prediction > 0.5 else "🎉Congratulations! No Tumor Detected"
            st.success(f"{result}")


elif page == 'Tumor Segmentation':
    st.title('🟥 Tumor Segmentation')
    st.write('Upload a brain MRI scan to highlight the tumor area.')

    uploaded_file = st.file_uploader('Upload MRI Image', type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], key="seg")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner('🔍 Segmenting tumor...'):
            img = image.resize((256, 256))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            model = tf.keras.models.load_model('models/Brain_Tumor_Segmentation.h5', compile=False)
            pred_mask = model.predict(img_array)[0]

            threshold = 0.05
            pred_mask = (pred_mask > threshold).astype(np.uint8) * 255
            mask_img = Image.fromarray(np.squeeze(pred_mask)).resize(image.size)
            tumor_pixels = np.sum(pred_mask > 0)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧠 Original MRI")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("### 🎯 Tumor Mask")
            st.image(mask_img, use_container_width=True)
