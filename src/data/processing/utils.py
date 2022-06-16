import pandas as pd
import numpy as np
from PIL import Image
import os
import pywt

def get_vector_from_data(imagen, iterations):
    data = imagen.flatten()
    return pywt.wavedecn(data=data, wavelet='haar', mode='symmetric', level=iterations)[0] 
    
def get_data_wavelet(path_dir, iterations,newshape=(64, 64)):

    x = []
    image_names = []

    for train_img in os.listdir(path_dir):
        image_path = f"{path_dir}\\{train_img}"
        img = Image.open(image_path)
        img = img.resize(newshape)
        img_np = np.asarray(img)
        vector_caracteristico = get_vector_from_data(img_np, iterations)
        x.append(vector_caracteristico)
        image_names.append(train_img)
    return np.asarray(x), np.asarray(image_names)