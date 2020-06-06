import cv2 as cv
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim


def photo_diff(image1, image2):
    if image1.shape != image2.shape:
        raise ArithmeticError("Images must have the same shape")

    image1 = (image1 * 255).astype(np.uint8)
    image2 = (image2 * 255).astype(np.uint8)

    image1_g = cv.cvtColor(image1, cv.COLOR_RGB2GRAY)
    image2_g = cv.cvtColor(image2, cv.COLOR_RGB2GRAY)

    score = ssim(image1, image2, multichannel=True)

    diff_image = abs(image1_g - image2_g)

    return (np.average(diff_image ** 2), score)


