import streamlit as st
from streamlit_extras.let_it_rain import rain
from PIL import Image
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv, find_dotenv
import os
from keras.models import load_model
from pymongo import MongoClient
import boto3
import streamlit as st
from streamlit_extras.let_it_rain import rain
from PIL import Image
import tensorflow as tf  
import random
from keras.preprocessing.image import img_to_array
import tempfile


# Set up directory in a system-independent way
model_dir = tempfile.gettempdir()

local_model1_path = os.path.join(model_dir, 'vgg19.h5')
local_model2_path = os.path.join(model_dir, 'xception.h5')

def download_model_from_s3(bucket_name, model_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, model_key, local_path)

# Download models if not downloaded yet
if not os.path.exists(local_model1_path):
    download_model_from_s3('introductiontoaiprojectmodels', 'vgg19.h5', local_model1_path)
if not os.path.exists(local_model2_path):
    download_model_from_s3('introductiontoaiprojectmodels', 'xception.h5', local_model2_path)

def try_load_model(path):
    try:
        model = load_model(path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

model1 = try_load_model(local_model1_path)
model2 = try_load_model(local_model2_path)

# Logo path
logo_path = "Got U logo.jpg"

# Using column layout to center the logo
col1, col2, col3 = st.columns([1,1,1])
with col2:
    st.image(logo_path, use_column_width=True)



def classify_image(image, model):
    img = image.resize((224, 224))  # Resize image to model's expected input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image

    predictions = model.predict(img_array) 
    return predictions
    
"""Function to classify the image as real or fake based on our model."""

    # This is a temporary example of image tuning for the model:


# Streamlit app layout
st.title('Welcome to Got U: Your Go-To Deep Fake Image Detector 🕵🏻')
st.divider()

st.write("""
Navigating the digital world's real vs. fake landscape just got easier! With "Got U," you're one upload away from uncovering the truth behind any face image. 
**Why "Got U"?**
- **Spot the Real Deal**: Instantly find out if that image is genuine or a clever fake.
- **Simplicity is Key**: Our straightforward design means you get results fast, no tech wizardry required.
- **Join the Truth Squad**: Help us fight the good fight against digital deception by identifying deepfakes.
""")

st.write("""
**Your Voice Matters**
Got feedback? We're all ears! Your insights help us make "Got U" even better, ensuring we stay on the frontline of deepfake detection.
""")

st.write("😁 **Thanks for teaming up with Got U. Let's keep it real together!**")

st.divider()

st.subheader("Now, let's get into action...")
st.text("Please upload or drag and drop your image")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    prediction1 = classify_image(image, model1)
    prediction2 = classify_image(image, model2)
    
    prediction = (prediction1 + prediction2)/2


    if prediction < 0.5:     
        st.error("The image is likely Fake ☹️👎🏻")
        rain(
        emoji="💀",
        font_size=40,
        falling_speed=3,
        animation_length=[5, 'seconds'], 
    )
    else:
        st.success("The image is likely Real 😁👍🏻")
        st.balloons()