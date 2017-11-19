import os
import numpy as np  
import cv2

#preprocessing module (normalizes image size and shape to square, converts image color to grayscale)
def preprocess_image(image):
    image = cv2.imread(image)
    image = cv2.resize(image, (35,35))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

