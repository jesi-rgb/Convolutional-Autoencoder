import json
import cv2
import numpy as np
import os
from multiprocessing import Pool

def pre_process_images(img_path, downscaling_factor=8, color_mode=1):
    img = cv2.imread(img_path, 1)[342:,:]
    aspect_ratio = img.shape[0] / img.shape[1]

    if(color_mode == 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    final_shape = (img.shape[0] // downscaling_factor, int((img.shape[0] // downscaling_factor) * (1/aspect_ratio)))
    img = img / 255  # it is very important to normalize the pixel values
    return cv2.resize(img, (final_shape[1], final_shape[0]))

def prepare_dataset(images_list, data_path, color_mode=1):
    print("Processing", len(images_list), "images")


    os.makedirs("serialized/"+data_path.split("/")[-1], exist_ok=True)
    
    pool = Pool(8)

    division1 = 6 * len(images_list) // 10
    division2 = 8 * len(images_list) // 10
    
    train_paths = []
    for img_path in images_list[:division1]:
        train_paths.append(os.path.join(data_path, img_path))
    
    x_train = np.array(pool.map(pre_process_images, train_paths), dtype=np.float32)
    serialize_data(x_train, "serialized/"+data_path.split("/")[-1]+"/x_train.npy")
    print("Finished processing x_train")


    test_paths = []
    for img_path in images_list[division1:division2]:
        test_paths.append(os.path.join(data_path, img_path))
    
    x_test = np.array(pool.map(pre_process_images, test_paths), dtype=np.float32)
    serialize_data(x_test, "serialized/"+data_path.split("/")[-1]+"/x_test.npy")
    print("Finished processing x_test")

    
    predict_paths = []
    for img_path in images_list[division2:-1]:
        predict_paths.append(os.path.join(data_path, img_path))
    
    x_predict = np.array(pool.map(pre_process_images, predict_paths), dtype=np.float32)
    serialize_data(x_predict, "serialized/"+data_path.split("/")[-1]+"/x_predict.npy")
    print("Finished processing x_predict")

    pool.close()
    pool.join()

    return x_train, x_test, x_predict

def serialize_data(np_array, file_name):
    np.save(file_name, np_array)
    
    print("Finished serializing", file_name)


def deserialize_data(loc_name):
    x_train = np.load("serialized/"+loc_name+"/x_train.npy")
    x_test = np.load("serialized/"+loc_name+"/x_test.npy")
    x_predict = np.load("serialized/"+loc_name+"/x_predict.npy")

    print("Preprocessed images loaded from disk")
    return x_train, x_test, x_predict


def obtain_dataset(images_list, data_path):
    if(os.path.exists("serialized/"+data_path.split("/")[-1]) and len(os.listdir("serialized/"+data_path.split("/")[-1])) > 0):
        x_train, x_test, x_predict = deserialize_data(data_path.split("/")[-1])
        print("No preprocessing needed, we got serialized data")
    else:
        print("No serialized data found, lets pre process it and serialize it")
        x_train, x_test, x_predict = prepare_dataset(images_list, data_path, color_mode=1)

    return x_train, x_test, x_predict 