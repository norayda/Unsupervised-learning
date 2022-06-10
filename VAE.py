import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

class VAE:

    def __init__(self, layers, X: np.ndarray, validation_x, activation,latent_dim):
        self.X =X

        self.validation_x = validation_x
        self.latent_dim = latent_dim
        self.decoder = keras.Sequential()
        self.encoder = keras.Sequential()


        means,logvar = self._init_encoder(layers,activation)
        output = self.Sampling(means,logvar)
        self._init_decoder(layers,activation,output)
        input_tensor = keras.layers.Input(shape = self.X[0].shape)
        samplingout = keras.layers.Lambda(lambda x: self.Sampling(*x))(self.encoder(input_tensor))
        self.model = keras.Model(input_tensor, self.decoder(samplingout))

    def _init_encoder(self, layers, activation):
        encode_layers = []
        input_tensor =keras.layers.Input(shape = self.X[0].shape)
        #encode_layers.append(keras.layers.Reshape((28, 28)))
        shape = [self.X.shape[-i] for i in range(1,len(self.X.shape))]
        shape = reduce(operator.mul, shape, 1)
        encode_layers.append(keras.layers.Flatten())
        for i, l in enumerate(layers):
            encode_layers.append(keras.layers.Dense(l, activation=activation, input_shape = (layers[i-1],))) #
            #encode_layers.append(keras.layers.Reshape((28,28)))
        output = keras.Sequential(encode_layers)(input_tensor)
        means = tf.keras.layers.Dense(self.latent_dim, activation="linear", name="means")(input_tensor)
        logvar = tf.keras.layers.Dense(self.latent_dim, activation="linear", name="logvar")(input_tensor)
        #encode_layers.append(keras.layers.Dense(self.latent_dim, activation=activation,input_shape=[l])) #La dernière couche est
        self.encoder = keras.Model(input_tensor, [means, logvar])

        return means,logvar


    def _init_decoder(self, layers, activation,output):
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

    def Sampling(self,means,logvar): ###
        batch = tf.shape(means)[0]
        dim = tf.shape(means)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return means + tf.exp(0.5 * logvar) * epsilon

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
    esp_latent = 10
    model = VAE(layers, x_train,x_test, 'relu',esp_latent)
    model.fit(2, 0.01,256)

    id_test = 15 #Indice de l'image à tester

    X_ = X[id_test]
    #X_ = tf.cast(X_, tf.float32)
    encoded = model.encoder(np.array([X_]))
    decoded = model.decoder(np.array(encoded))[0]

    img =tf.reshape(decoded,(28,28))

    #plt.imshow(img)
    #plt.gray()
    #plt.show()

    fig, (ax1,ax2) = plt.subplots(1,2)
    plt.gray()
    ax1.imshow(X[id_test])
    ax2.imshow(img)

    plt.show()