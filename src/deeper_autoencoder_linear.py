# Code taken from https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Flatten, Reshape, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, model_from_json
# from keras import backend as K

def build_autoencoder(shape):
    input_layer = Input(shape=shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    encoded = Dense(2304, activation='relu')(x)
    
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    decoded = Conv2D(shape[len(shape)-1], (3, 3), activation='linear', padding='same')(x)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

def save_autoencoder(model, location, weights_loc):
    # serialize model to JSON
    model_json = model.to_json()
    with open(location, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_loc)
    print("Saved model to disk")

def load_autoencoder(model_path, weights_path):
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    loaded_model.compile(optimizer='adam', loss='binary_crossentropy')
    print("Loaded model from disk")
    return loaded_model

