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
import imutils
import cv2
from face_detection_3 import detect_face

st.set_page_config(layout="wide", page_title="Got U DeepFake")

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

@st.cache(allow_output_mutation=True)
def load_model_from_path(model_path):
    model = load_model(model_path)
    return model

model1 = load_model_from_path(local_model1_path)
model2 = load_model_from_path(local_model2_path)


# load_dotenv(find_dotenv())                      # load environment file to use password saved as an evironment var
# password = os.environ.get("MONGODB_PWD")        # assing password store in env var

#connection string
MONGODB_URI = f"mongodb+srv://gotudeepfake:lambton3014@cluster0.jpdo5rg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(MONGODB_URI)

db = client.deepfake                            # DataBase
collect = db.deepfake_report                     # Collection to stores counter for report

# Logo path
logo_path = "logo-gotudeepfake.png"
fake_icon = "fake.png"
real_icon = "real.png"


# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((2, 3))

cola1, cola2, cola3, cola4, cola5, cola6, cola7, cola8, cola9 = st.columns([1,1,1,1,1,1,1,1,1])


# This is a tempral code to test the web app
def classify_image(image, model):
    img = image.resize((224, 224))  # Resize image to model's expected input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image

    predictions = model.predict(img_array) 
    return predictions

    '''
    # Load directly the model
    model = load_model('./deepfake-detection-project/deploy/model_epoch_10_val_accuracy_1.0000.h5')
    
    # Get the model's prediction
    prediction = model.predict(image)
    predicted_class = 'Real' if prediction[0][0] > 0.5 else 'Fake'
    print(f'Prediction: {prediction[0][0]}')
    print(f'Predicted class: {predicted_class}')
'''

def rezize_image(img):

    """Function to resize the image dimension."""
    #img = image.resize((224, 224))  # Resize image to model's expected input size

   
    # Load the image and convert it to a numpy array
    target_size=(224, 224)
    image = img.resize(target_size)  # Resize the image to the specified target size

    img_array = np.array(image)  # Convert the resized image to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    img_array = img_array / 255.0  # Rescale the image

    return img_array

def classify(image):

    #st.write("Classifying...")
    
    prediction1 = classify_image(image, model1)
    prediction2 = classify_image(image, model2)
    
    prediction = (prediction1 + prediction2)/2
    
    # Here the model will give the probability to make the prediction
    # 0.5 is an example, threshold can be adjusted
    predicted_class = 'Real' if prediction > 0.5 else 'Fake'
    print(f'Prediction Average: {prediction}') 
    print(f'Predicted class Average: {predicted_class}')

    if predicted_class == "Fake":     
        #st.error("The image is likely Fake ☹️👎🏻")
        rain(
        emoji="💀",
        font_size=40,
        falling_speed=3,
        animation_length=[5, 'seconds'], 
    )
        st.image(fake_icon,width=70)
        compute_report(0,1)

    else:
        #st.success("The image is likely Real 😁👍🏻")
        st.balloons()
        st.image(real_icon,width=70)
        compute_report(1,0)

    
def compute_report(real, fake):   

    report  = collect.find_one()
    print(report)
    qry = { "_id": report["_id"] }

    submitted = int(report["number_submitted"]) + 1
    real_image_count = int(report["real_images_caught"]) + real
    fake_image_count = int(report["fake_images_caught"]) + fake

    with row1_2:
         col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.write("Number of time submitted")
        st.markdown(f"{submitted}")

    with col2:
        st.write("Real images caught")
        st.write(f"\t{real_image_count}")

    with col3:
        st.write("fake images caught")
        st.write(f"{fake_image_count}")


    deepfake_report= { "$set": {
        "number_submitted":submitted ,
        "real_images_caught":real_image_count,
        "fake_images_caught":fake_image_count
    }}

    collect.update_one(qry, deepfake_report, upsert=True)


# def classify_image(image):
#     """Function to classify the image as real or fake based on our model."""
    
#     # This is a temporary example of image tuning for the model:
#     img = image.resize((224, 224))  # Resize image to model's expected input size
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create a batch

#     predictions = model.predict(img_array)
#     return predictions


with row1_1:
    colx1_1, colx1_2, colx1_3 = st.columns([1,1,1])
    with colx1_2:
        st.image(logo_path)
    
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

with row1_2:
    st.subheader("Now, let's get into action...")
    st.text("Please upload or drag and drop your image")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        
        #Call to method that use opencv model to face recognition
        if(detect_face(uploaded_image)==0):
            st.write(f'This is not a face image!!!!')

        else: # it is a face
            image = Image.open(uploaded_image)
            
            image_resize = rezize_image(image)
            col1, col2, col3 = st.columns([1,1,1])
            col1.empty()
            with col2:
                st.image(image_resize, caption='Uploaded Image')
                if st.button('Validate', type="primary"):
                    classify(image)
            col3.empty()

    