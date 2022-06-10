import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

class AutoEncoder:
    X: np.ndarray  # chaque ligne est une image
    # Y: int
    K: int
    encoder : tf.keras.Sequential()
    decoder : tf.keras.Sequential()
    model : tf.keras.Sequential()
    latent_dim : int
    validation_x: np.ndarray


    def __init__(self, layers, X: np.ndarray, validation_x, activation,latent_dim):
        self.X =X

        self.validation_x = validation_x
        self.latent_dim = latent_dim
        self.decoder = keras.Sequential()
        self.encoder = keras.Sequential()

        self._init_encoder(layers,activation)
        self._init_decoder(layers,activation)
        self.model = keras.Sequential([self.encoder, self.decoder])

    def _init_encoder(self, layers, activation):
        encode_layers = []
        #encode_layers.append(keras.layers.Reshape((28, 28)))
        shape = [self.X.shape[-i] for i in range(1,len(self.X.shape))]
        shape = reduce(operator.mul, shape, 1)
        encode_layers.append(keras.layers.Flatten())
        for i, l in enumerate(layers):
            encode_layers.append(keras.layers.Dense(l, activation=activation, input_shape = (layers[i-1],))) #
            #encode_layers.append(keras.layers.Reshape((28,28)))
        encode_layers.append(keras.layers.Dense(self.latent_dim, activation=activation,input_shape=[l])) #La dernière couche est
        self.encoder = keras.Sequential(encode_layers)

    def _init_decoder(self, layers, activation):
        decode_layers = []
        reverse_layer = layers[::-1]
        for i, l in enumerate(reverse_layer):
            decode_layers.append(keras.layers.Dense(l, activation=activation))
           # decode_layers.append(keras.layers.Reshape((28, 28)))


        shape = [self.X.shape[-i] for i in range(1,len(self.X.shape))]
        shape = reduce(operator.mul, shape, 1)

        decode_layers.append(keras.layers.Dense(shape, activation='sigmoid'))
        decode_layers.append(keras.layers.Reshape(self.X.shape[1:]))
        self.decoder = keras.Sequential(decode_layers)

    def fit(self, epochs, lr,batch_size):
        loss = tf.keras.losses.MeanSquaredError(reduction='sum')
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer,loss=loss,epochs=epochs,shuffle=True)
        self.model.fit(x=self.X, y=self.X ,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(self.validation_x, self.validation_x))



if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X =x_train

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    layers = [512,324,64]
    esp_latent = 100
    model = AutoEncoder(layers, x_train,x_test, 'relu',esp_latent)
    model.fit(10, 0.01,256)

    id_test = 6 #Indice de l'image à tester

    encoded = model.encoder(np.array([X[id_test]]))
    decoded = model.decoder(np.array(encoded))

    #plt.imshow(img)
    #plt.gray()
    #plt.show()

    fig, (ax1,ax2) = plt.subplots(1,2)
    plt.gray()
    ax1.imshow(X[id_test])
    ax2.imshow(decoded)

    plt.show()





