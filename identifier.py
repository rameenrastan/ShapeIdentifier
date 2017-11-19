import os
import tflearn
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.conv import conv_2d, max_pool_2d
import numpy as np  
import cv2
from pathlib import Path
from tflearn.layers.estimator import regression

#preprocessing module (normalizes image size and shape to square, converts image color to grayscale)
def preprocess_image(image):
    image = str(image)
    image = cv2.imread(image)
    image = cv2.resize(image, (35, 35))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

#feature extraction module (preprocesses and converts image to numpy array of pixel color values)
def extract_features(image):
    image = preprocess_image(image)
    image_feature_data = np.array(image)
    return image_feature_data

#creates training data through preprocessing and feature extraction of training dataset
def create_leaf_training_data():
    feature_training_dataset = []
    for image in os.listdir('leafs/'): 
        image = 'leafs/' + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, 1])
    return feature_training_dataset    

training_feature_dataset = create_leaf_training_data() 
np.save('training_feature_dataset.npy', training_feature_dataset)

#define our convolutional neural network model using tensorflow and tflearn
def create_cnn_model():
    cnn = input_data(shape=[None, 35, 35, 1], name='input')
    cnn = conv_2d(cnn, 32, 2, activation = "relu")
    cnn = max_pool_2d(cnn, 2)
    cnn = conv_2d(cnn, 64, 2, activation = "relu")
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 2, activation="softmax")
    cnn = regression(cnn, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy")
    return cnn

#instatiate our model
cnn_model = tflearn.DNN(create_cnn_model(), tensorboard_dir = "cnn")

