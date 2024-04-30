# DeepFake_detection

This project focuses on developing a deep learning model to identify deepfake images, specifically targeting manipulated face images from https://www.thispersondoesnotexist.com/

Utilizing TensorFlow and the Xception/VGG19 architecture pre-trained on ImageNet, the model has been trained to distinguish between 'real' and 'fake' images. The model benefits from enhancements like image preprocessing, data augmentation, and performance tuning with a focus on improving the accuracy and robustness against deepfake technologies.

Models were integrated to make a final prediction, averaging the probabilities getting one last call to determine the nature of the images.

User interface was developed with Streamlit, models where stored in an AWS S3 bucket and called by the model to make the predictions.
