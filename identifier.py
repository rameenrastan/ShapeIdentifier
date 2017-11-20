import os
import tensorflow as tf
import tflearn
from tflearn.layers.core import dropout, fully_connected, input_data
import numpy as np  
import cv2
from tflearn.layers.conv import conv_2d, max_pool_2d
from pathlib import Path
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

#shape labels
square = [1, 0, 0, 0]
circle = [0, 1, 0, 0]
triangle = [0, 0, 1, 0]
star = [0, 0, 0, 1]

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
def create_labeled_feature_dataset():
    feature_training_dataset = []
    for image in os.listdir('data/square/'): 
         image = 'data/square/' + str(image)
         features = extract_features(image)
         feature_training_dataset.append([features, np.array(square)])
    for image in os.listdir('data/circle/'): 
         image = 'data/circle/' + str(image)
         features = extract_features(image)
         feature_training_dataset.append([features, np.array(circle)])   
    for image in os.listdir('data/star/'): 
         image = 'data/star/' + str(image)
         features = extract_features(image)
         feature_training_dataset.append([features, np.array(star)])    
    for image in os.listdir('data/triangle/'): 
         image = 'data/triangle/' + str(image)
         features = extract_features(image)
         feature_training_dataset.append([features, np.array(triangle)])          

    #iterate over leaf folder (contains leaf images)
    # for image in os.listdir('leafs/'): 
    #     image = 'leafs/' + str(image)
    #     features = extract_features(image)
    #     feature_training_dataset.append([features, np.array([1])])
    # #iterate over other folder (contains non-leaf images)    
    # for image in os.listdir('other/'): 
    #     image = 'other/' + str(image)
    #     features = extract_features(image)
    #     feature_training_dataset.append([features, np.array([0])])   
    #shuffles the numpy array (randomizes the order)   


    np.random.shuffle(feature_training_dataset)     
    return feature_training_dataset    

training_feature_dataset = create_labeled_feature_dataset() 
np.save('training_feature_dataset.npy', training_feature_dataset)

#define our convolutional neural network model using tensorflow and tflearn
def create_cnn_model():
    cnn = input_data(shape=[None, 35, 35, 1], name="input")
    cnn = conv_2d(cnn, 32, 2, activation = "relu")
    cnn = max_pool_2d(cnn, 2)
    cnn = conv_2d(cnn, 64, 2, activation = "relu")
    cnn = max_pool_2d(cnn, 2)
    cnn = fully_connected(cnn, 1024, activation = "relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation = "relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation = "relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation = "relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation = "relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation = "relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation = "relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation = "relu")
    cnn = fully_connected(cnn, 4, activation = "softmax")
    cnn = regression(cnn, optimizer = "adam", learning_rate = 0.001, loss = "categorical_crossentropy")
    return cnn
    
#instantiate our model
cnn = tflearn.DNN(create_cnn_model(), tensorboard_dir = "cnn")

#checks if modeled was already trained and saved
training_data = training_feature_dataset[:-500]
testing_data = training_feature_dataset[-500:]

x = np.array([i[0] for i in training_data]).reshape(-1, 35, 35, 1)
y = np.array([i[1] for i in training_data])

test_x = np.array([i[0] for i in testing_data]).reshape(-1, 35, 35, 1)
test_y = np.array([i[1] for i in testing_data])

cnn.fit(x, y, n_epoch=10, validation_set=(test_x,test_y), snapshot_step=500, show_metric=True, run_id='leaf')

cnn.save('cnn.tflearn')

image = 'data/circle/1.png'
image_features = extract_features(image)
data = image_features.reshape(35, 35, 1)
prediction = cnn.predict([data])
print(np.argmax(prediction))
