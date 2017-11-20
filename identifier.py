import os
import tensorflow as tf
import tflearn
from tflearn.layers.core import dropout, fully_connected, input_data
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
    cnn = input_data(shape=[None, image_size, image_size, 1], name="input")
    cnn = conv_2d(cnn, 32, 2, activation="relu")
    cnn = max_pool_2d(cnn, 2)
    cnn = conv_2d(cnn, 64, 2, activation="relu")
    cnn = max_pool_2d(cnn, 2)
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 512, activation="relu")
    cnn = fully_connected(cnn, 4, activation="softmax")
    cnn = regression(cnn, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy")
    return cnn

# instantiate our model using tflearn
cnn = tflearn.DNN(create_cnn_model(), tensorboard_dir="cnn")

#if our trained model already exists, load it
if os.path.exists(model + '.meta'):
    cnn.load(model)
#if our trained model doesn't exist, train the model and save it    
else:
    training_data = training_feature_dataset[:-250]
    testing_data = training_feature_dataset[-250:]

    x = np.array([i[0] for i in training_data]).reshape(-1, image_size, image_size, 1)
    y = np.array([i[1] for i in training_data])

    test_x = np.array([i[0] for i in testing_data]).reshape(-1, image_size, image_size, 1)
    test_y = np.array([i[1] for i in testing_data])

    cnn.fit(x, y, n_epoch=3, validation_set=(test_x, test_y), snapshot_step=500, show_metric=True, run_id=model)

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
