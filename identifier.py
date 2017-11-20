import os
import tensorflow as tf
import tflearn
from tflearn.layers.core import dropout, fully_connected, input_data
import numpy as np
import cv2
from tflearn.layers.conv import conv_2d, max_pool_2d
from pathlib import Path
from tflearn.layers.estimator import regression

# import matplotlib.pyplot as plt

# shape labels
square = [1, 0, 0, 0]
circle = [0, 1, 0, 0]
triangle = [0, 0, 1, 0]
star = [0, 0, 0, 1]
img_size = 35
LR = 1e-3
MODEL_NAME = 'shapeClassifer-{}-{}.model'.format(LR, '2conv-basic')

TRAIN_DIR = 'data/'
SQUARE_DIR = TRAIN_DIR + "square/"
CIRCLE_DIR = TRAIN_DIR + "circle/"
STAR_DIR = TRAIN_DIR + "star/"
TRIANGLE_DIR = TRAIN_DIR + "triangle/"


# preprocessing module (normalizes image size and shape to square, converts image color to grayscale)
def preprocess_image(image):
    image = str(image)
    image = cv2.imread(image)
    image = cv2.resize(image, (img_size, img_size))
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
    for image in os.listdir(SQUARE_DIR):
        image = SQUARE_DIR + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, np.array(square)])
    for image in os.listdir(CIRCLE_DIR):
        image = CIRCLE_DIR + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, np.array(circle)])
    for image in os.listdir(STAR_DIR):
        image = STAR_DIR + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, np.array(star)])
    for image in os.listdir(TRIANGLE_DIR):
        image = TRIANGLE_DIR + str(image)
        features = extract_features(image)
        feature_training_dataset.append([features, np.array(triangle)])

    np.random.shuffle(feature_training_dataset)
    return feature_training_dataset


training_feature_dataset = create_labeled_feature_dataset()
np.save('training_feature_dataset.npy', training_feature_dataset)


# define our convolutional neural network model using tensorflow and tflearn
def create_cnn_model():
    cnn = input_data(shape=[None, img_size, img_size, 1], name="input")
    cnn = conv_2d(cnn, 32, 2, activation="relu")
    cnn = max_pool_2d(cnn, 2)
    cnn = conv_2d(cnn, 64, 2, activation="relu")
    cnn = max_pool_2d(cnn, 2)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = dropout(cnn, 0.8)
    cnn = fully_connected(cnn, 1024, activation="relu")
    cnn = fully_connected(cnn, 4, activation="softmax")
    cnn = regression(cnn, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy")
    return cnn


# instantiate our model
cnn = tflearn.DNN(create_cnn_model(), tensorboard_dir="cnn")

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    cnn.load(MODEL_NAME)
    print('model loaded!')
else:
    # checks if modeled was already trained and saved
    training_data = training_feature_dataset[:-500]
    testing_data = training_feature_dataset[-500:]

    x = np.array([i[0] for i in training_data]).reshape(-1, img_size, img_size, 1)
    y = np.array([i[1] for i in training_data])

    test_x = np.array([i[0] for i in testing_data]).reshape(-1, img_size, img_size, 1)
    test_y = np.array([i[1] for i in testing_data])

    cnn.fit(x, y, n_epoch=3, validation_set=(test_x, test_y), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    cnn.save(MODEL_NAME)

image = 'data/circle/1.png'
image_features = extract_features(image)
data = image_features.reshape(35, 35, 1)
prediction = cnn.predict([data])

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
