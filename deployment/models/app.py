from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
from tensorflow.keras.preprocessing import image
#system level operations (like loading files)
import sys
import io

#tell to app where the saved model is
#print("model path",sys.path.append(os.path.abspath("./model")))

app = Flask(__name__)
model = None

#Xception
def model1_load():
        # load the pre-trained model (here we are using a model
        global model1
        model1 = load_model('./model/Xception/model_epoch_11_val_accuracy_0.9414.h5')
#VGG19
def model2_load():
        # load the pre-trained model (here we are using a model
        global model2
        model2 = load_model('./model/VGG19/model_epoch_12_val_accuracy_0.8990.h5')


def preprocess_image(image, target=(224, 224)):

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(target)
        image_array = np.array(image) / 255.0  # Scale pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        return image_array

@app.route('/')
def index_view():
        return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']

        if file:
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            
            prediction1 = model1.predict(processed_image) # probability
            prediction2 = model2.predict(processed_image) # probability

            # Convert prediction to desired format/response
            # assuming a classification model:
            predicted_class1 = np.argmax(prediction1, axis=1)
            predicted_class2 = np.argmax(prediction2, axis=1)

            # print prediction model1
            predictedClass1 = 'Real' if prediction1[0][0] > 0.5 else 'Fake'
            print(f'Prediction1: {prediction1[0][0]}')
            print(f'Predicted class1: {predictedClass1}')

             # print prediction model2
            predictedClass2 = 'Real' if prediction2[0][0] > 0.5 else 'Fake'
            print(f'Prediction2: {prediction2[0][0]}')
            print(f'Predicted class2: {predictedClass2}')
                  
           
            return jsonify({'prediction1': float(prediction1[0][0]),'prediction2': float(prediction2[0][0])})

            #return jsonify({'prediction': int(predicted_class[0])})
        else:
            return jsonify({'error': 'Error processing file'}), 500
        
        
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
		"please wait until server has fully started"))
    model1_load()
    model2_load()
    app.run()