import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras

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
        encode_layers.append(keras.layers.Flatten(input_shape=(784,)))
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
        decode_layers.append(keras.layers.Dense(784))
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

    def call_encode(self,x):
        return self.encoder(x)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X =x_train

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    layers = [52, 16, 8]
    esp_latent = 2
    model = AutoEncoder(layers, x_train,x_test, 'relu',esp_latent)
    model.fit(10, 0.01,256)


    encoded = model.call_encode(np.array([X[4]]))
    decoded = model.decoder(np.array(encoded))

    img = np.reshape(decoded,(28,28))
    #plt.imshow(img)
    #plt.gray()
    #plt.show()

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(X[4])
    ax2.imshow(img)

    plt.gray()
    plt.show()





