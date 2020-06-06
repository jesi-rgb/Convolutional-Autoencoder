import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from preprocessing import pre_process_images
from deeper_autoencoder_trans_seven import load_autoencoder
import matplotlib.pyplot as plt
import numpy as np
from photo_diff import photo_diff
import argparse
import pickle

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="model")
    ap.add_argument("-i", "--image", required=True,
        help="input image")
    ap.add_argument("-c", "--color", required=True,
        help="color format")
    args = vars(ap.parse_args())

    model_path = args["model"]

    model = load_autoencoder(("model/" + model_path), ("model/" + model_path + "_w"))
    decisionTree = pickle.load(open("model/decision_tree.sav", 'rb'))

    image_path = args["image"]
    color_mode = int(args["color"])

    processed_img = pre_process_images(image_path, color_mode=color_mode)

    if(color_mode != 0):
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(processed_img)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)

        predicted_image = model.predict(np.array([processed_img]))
        print("Model evaluation:", model.evaluate(x = np.array([processed_img]), y = np.array([processed_img])))
        data = photo_diff(processed_img, predicted_image[0])
        print(f"Image difference: Average error: {data[0]} || SSIM: {data[1] * 100:.2f}%")
        print(f"Predicted label: {decisionTree.predict(np.array([data[0], data[1]]).reshape(1, -1))}")

        ax[1].imshow(predicted_image[0])
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(processed_img, cmap='gray')
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)

        processed_img = processed_img.reshape([processed_img.shape[0], processed_img.shape[1], 1])
        predicted_image = model.predict(np.array([processed_img]))

        ax[1].imshow(predicted_image[0].reshape(predicted_image.shape[1], predicted_image.shape[2]), cmap='gray')
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.show()
        
    exit()