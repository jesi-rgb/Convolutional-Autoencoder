import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from deeper_autoencoder import load_autoencoder
import numpy as np

model = load_autoencoder(("model/autoencoder_base_t_deeper"), ("model/autoencoder_base_t_deeper_w"))

processed

prediction = model.predict(processed_array)

# print(prediction)

