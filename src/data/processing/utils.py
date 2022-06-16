import pandas as pd
import numpy as np
from PIL import Image
import os
import pywt

def get_vector_from_data(imagen, iterations):

    LL, (LH, HL, HH) = pywt.dwt2(imagen, 'haar')
    for _ in range(iterations - 1):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL.flatten()
    
def get_data_wavelet(path_dir, iterations, width=100, height=100):

    x = []
    image_names = []

    for train_img in os.listdir(path_dir):
        image_path = f"{path_dir}\\{train_img}"
        img = Image.open(image_path)
        newsize = (width, height)
        img = img.resize(newsize)
        vector_caracteristico = get_vector_from_data(img, iterations)
        x.append(vector_caracteristico)
        image_names.append(train_img)
    return np.asarray(x), np.asarray(image_names)