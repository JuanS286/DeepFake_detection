# Got U Deepfake Detection System

Welcome to the Got U Deepfake Detection System repository. This project aims to provide an effective solution for detecting deepfake images using advanced deep learning techniques. 

## Project Overview

The Got U Deepfake Detection System leverages convolutional neural network (CNN) models to identify deepfake images, which are synthetic images generated using artificial intelligence. The project employs pre-trained models such as Xception and VGG19 to extract intricate features from images, ensuring robust and accurate deepfake detection.

## Problem Statement

The increasing sophistication and accessibility of deepfake technology pose significant risks, including misinformation, reputation damage, and erosion of trust in digital media. Our goal is to develop reliable tools for detecting deepfake images to maintain the authenticity and trustworthiness of visual content shared online.

## Project Objectives

1. **Develop a Deepfake Detection Model**: Using pre-trained CNN architectures (Xception and VGG19) to build a robust model for detecting deepfake images.
2. **Enhance Model Interpretability**: Employ LIME (Local Interpretable Model-agnostic Explanations) to understand how models make decisions and to improve their transparency.
3. **Deploy the Model in a Production Environment**: Create a user-friendly interface using Streamlit and deploy the system on the Streamlit Community Cloud with models stored in AWS S3.

## Features

- **Advanced Deep Learning Models**: Utilizes Xception and VGG19 architectures for effective feature extraction and classification.
- **Data Augmentation and Normalization**: Enhances model performance by diversifying the training dataset and ensuring consistent input scaling.
- **Model Interpretability**: Uses LIME to provide insights into the decision-making process of the models.
- **User-Friendly Interface**: Built with Streamlit to allow easy image uploads and display results interactively.
- **Deployment on Cloud**: Leverages AWS S3 for model storage and Streamlit Community Cloud for deployment, ensuring scalability and accessibility.

## Data Processing Pipeline

1. **Data Collection**: Gathered real and fake images for training, validation, and testing.
2. **Data Preprocessing**: Resized images to 224x224 pixels, normalized pixel values, and applied data augmentation techniques.
3. **Feature Extraction**: Used Xception and VGG19 models to extract meaningful features from the images.
4. **Model Training**: Trained the models on a balanced dataset to ensure generalization and prevent overfitting.
5. **Model Evaluation**: Assessed model performance using metrics such as accuracy, precision, recall, F1 score, and AUC.
6. **Model Interpretability**: Applied LIME to understand model predictions and improve interpretability.
7. **Deployment**: Deployed the system using Streamlit for the UI and AWS S3 for model storage.

## Model Performance

- **VGG19**:
  - Training Accuracy: 0.9293
  - Validation Accuracy: 0.8990
  - Test Accuracy: 0.8967
  - Precision: 0.8839
  - Recall: 0.9133
  - F1 Score: 0.8984
  - AUC: 0.9605

- **Xception**:
  - Training Accuracy: 0.8467
  - Validation Accuracy: 0.8850
  - Test Accuracy: 0.8850
  - Precision: 0.8427
  - Recall: 0.9467
  - F1 Score: 0.8917
  - AUC: 0.9500

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/deepfake-detection-project.git
   cd deepfake-detection-project
   ```

2. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit App**:

```bash
streamlit run app.py
```

4. **Upload Images for Detection**: Use the Streamlit interface to upload images and get real-time feedback on whether they are real or fake.

## Deployment

The application is deployed on the Streamlit Community Cloud and utilizes AWS S3 for storing the deep learning models. This setup ensures efficient handling of large model files and provides a seamless user experience.

