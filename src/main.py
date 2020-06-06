import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import preprocessing as prep
import deeper_autoencoder as d_cae
import datetime, random

def plot_hist(history):
    try:
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower left')
        plt.savefig("media/jason_aec_1200.pdf")
    except:
        pass

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower left')
    plt.savefig("media/overnight_plot_"+str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))+".pdf")

def plot_imgs(originals, predictions):    
    n = 5
    fig, ax = plt.subplots(2, n)
    for i in range(n):
        # display original
        # ax[0].set_title("Original Images")
        ax[0, i].imshow(originals[i])
        ax[0, i].get_xaxis().set_visible(False)
        ax[0, i].get_yaxis().set_visible(False)

        # display reconstruction
        # ax[1].set_title("Reconstructed Images")
        ax[1, i].imshow(predictions[i])
        ax[1, i].get_xaxis().set_visible(False)
        ax[1, i].get_yaxis().set_visible(False)


    plt.tight_layout()
    plt.savefig("media/trained_prediction_gallery_"+str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))+".pdf")
    # plt.show()

if __name__ == "__main__":

    data_path = "data/locations/68" 
    images_list = os.listdir(data_path)
    random.shuffle(images_list)

    x_train, x_test, _ = prep.obtain_dataset(images_list, data_path)

    print(x_train[0].shape)
    model = d_cae.load_autoencoder("model/autoencoder_dense", "model/autoencoder_dense_w")
    model.summary()

    history = model.fit(x_train, x_train,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=2)
    plot_hist(history)

    d_cae.save_autoencoder(model, "model/autoencoder_dense_v2", "model/autoencoder_dense_w_v2")

    _, _, x_predict = prep.obtain_dataset(images_list, data_path)
    decoded_imgs = model.predict(x_predict)
    plot_imgs(x_predict, decoded_imgs)