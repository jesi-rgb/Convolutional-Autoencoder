import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import matplotlib.pyplot as plt
import numpy as np

import preprocessing as prep
from deeper_autoencoder import load_autoencoder
from main import plot_imgs



if __name__ == "__main__":
    model = load_autoencoder("model/autoencoder_gray", "model/autoencoder_gray_w")
    model.summary()

    data_path = "data/locations/46"
    images_list = os.listdir(data_path)

    _, _, x_predict = prep.obtain_dataset(images_list, data_path)
    random.shuffle(x_predict)
    predicted_images = model.predict(x_predict)

    plot_imgs(x_predict, predicted_images)
