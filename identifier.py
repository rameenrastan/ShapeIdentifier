import os
import tensorflow as tf
import tflearn
from tflearn.layers.core import fully_connected, input_data, dropout
import numpy as np
import cv2
from tflearn.layers.conv import conv_2d, max_pool_2d
from pathlib import Path
from tflearn.layers.estimator import regression
import argparse

# shape labels
square = [1, 0, 0, 0]
circle = [0, 1, 0, 0]
triangle = [0, 0, 1, 0]
star = [0, 0, 0, 1]

#specifies image size (images get transformed to this size during preprocessing)
image_size = 35

#name of our convolutional neural network model when saving as file
model = 'shapeClassifer'

#dataset directory
dataset_directory = 'data/'
#shape dataset directories
square_directory = dataset_directory + "square/"
circle_directory = dataset_directory + "circle/"
star_directory = dataset_directory + "star/"
triangle_directory = dataset_directory + "triangle/"

#allows user to input an image in the command line using --image [image path]
def parse(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    arg = parser.parse_args()
    return arg.image

# preprocessing module (normalizes image size and shape to square, converts image color to grayscale)
def preprocess_image(image):
    image = str(image)
    image = cv2.imread(image)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# feature extraction module (preprocesses and converts image to numpy array of pixel color values)
def extract_features(image):
    image = preprocess_image(image)
    image_feature_data = np.array(image)
    return image_feature_data


# creates training data through preprocessing and feature extraction of training dataset
def create_labeled_feature_dataset():
    feature_training_dataset = []
    for image in os.listdir(square_directory):
        image = square_directory + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, np.array(square)])
    for image in os.listdir(circle_directory):
        image = circle_directory + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, np.array(circle)])
    for image in os.listdir(star_directory):
        image = star_directory + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, np.array(star)])
    for image in os.listdir(triangle_directory):
        image = triangle_directory + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, np.array(triangle)])
    np.random.shuffle(feature_training_dataset)
    return feature_training_dataset

#create our feature training dataset
training_feature_dataset = create_labeled_feature_dataset()

# define our convolutional neural network model using tensorflow and tflearn
def create_cnn_model():
    #input layer (used to input our training data)
    cnn = input_data(shape=[None, image_size, image_size, 1], name="input")
    #2d convolutional layer
    cnn = conv_2d(cnn, 32, 2, activation="relu")
    cnn = max_pool_2d(cnn, 2)
    #2d convolutional layer
    cnn = conv_2d(cnn, 64, 2, activation="relu")
    cnn = max_pool_2d(cnn, 2)
    #2d convolutional layer
    cnn = conv_2d(cnn, 64, 2, activation="relu")
    cnn = max_pool_2d(cnn, 2)
    #fully connected layer
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.5)
    #fully connected layer
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.5)
    #fully connected layer
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.5)
    #fully connected layer
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.5)
    #fully connected layer
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.5)
    #fully connected layer
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.5)
    #fully connected layer
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.5)
    #fully connected layer
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = fully_connected(cnn, 4, activation="softmax")
    #regression layer: default optimizer: adam, default loss function: categorical cross entropy
    cnn = regression(cnn)
    return cnn

# instantiate our model using tflearn
cnn = tflearn.DNN(create_cnn_model(), tensorboard_dir="cnn")

#if our trained model already exists, load it
if os.path.exists(model + '.meta'):
    cnn.load(model)
#if our trained model doesn't exist, train the model and save it    
else:
    training_dataset = training_feature_dataset[:-250]
    validation_dataset = training_feature_dataset[-250:]

    feature_list = np.array([feature_set[0] for feature_set in training_dataset]).reshape(-1, image_size, image_size, 1)
    label_list = np.array([label[1] for label in training_dataset])

    validation_feature_list = np.array([feature_set[0] for feature_set in validation_dataset]).reshape(-1, image_size, image_size, 1)
    validation_label_list = np.array([label[1] for label in validation_dataset])

    cnn.fit(feature_list, label_list, n_epoch=3, validation_set=(validation_feature_list, validation_label_list), snapshot_step=500, show_metric=True, run_id=model)

    cnn.save(model)

#input our trained model and image, and prints our the prediction of what shape the image is
def make_prediction(model, image):
    image_features = extract_features(image)
    image_features = image_features.reshape(35, 35, 1)
    prediction = model.predict([image_features])
    if (np.argmax(prediction) == 0):
        print('This is a square!')
    elif (np.argmax(prediction) == 1):
        print('This is a circle!')
    elif (np.argmax(prediction) == 2):
        print('This is a triangle!')
    elif (np.argmax(prediction) == 3):
        print('This is a star!')
    else:
        print('This is not a square, circle, triangle or square!')

#make prediction based on the image the user specified in command line (using --image [image path])
image = parse()
make_prediction(cnn, image)        
