import cv2
import numpy as np
import scipy.ndimage

def gaussian_stack(image):
    """
    Returns gaussian stack of image where result[0] is the lowest resolution
    """
    max_sigma = max(image.shape[0], image.shape[1])
    sigma = 1
    result = []
    while sigma <= max_sigma:
        result.append(scipy.ndimage.gaussian_filter(image, sigma))
        sigma *= 2
    return result[::-1]