import numpy as np
import random

from tensorflow.keras.datasets import mnist

class Kmeans:
    X: np.ndarray  # chaque ligne est une image
    # Y: int
    K: int
    stop: int
    centroids = []

    def __init__(self,X,K,stop):
        self.K=K
        self.X = X
        self.stop = stop

    def choose_centroids(self):
        """Choisi aléatoirement K centroides"""
        for i in range(self.K):
            self.centroids.append(self.X[random.randint(0, self.X.shape[0])])


    def new_centroids(self):
        new_centroids = np.zeros((len(self.centroids), *self.X.shape[1:])) #contiendra le nouveau centroides
        centroids_label =np.empty(len(self.X)) # Contiendra la liste des indices des centroides le plus proche
        for i, x in enumerate(self.X):
            distances = np.empty(len(self.centroids))
            for j, c in enumerate(self.centroids):
                distances[j] = np.linalg.norm(x - c) #calcule la distane d'un point à un centroide
            centroids_label[i] = np.argmin(distances) #donne l'indice du centroide le plus proche de ce point x
        for i, k in enumerate(range(self.K)):
            group_by_k = np.where(centroids_label == k) # récupère les indices des points qui ont le centroide k plus proche
            new_centroids[i] = np.mean(self.X[group_by_k], axis=0) #prend la moyenne des points qui ont le centroid k comme plus proche centroide

        return new_centroids

    def condition_finale(self, centroids, old_centroids):
        return (centroids == old_centroids).all()

    def fit(self):
        update = True
        centroids = self.choose_centroids()
        count = 1
        while update:
            print(" epoch :",count,"sur ",self.stop)
            old_centroids = np.copy(centroids)
            centroids= self.new_centroids()
            if self.condition_finale(centroids,old_centroids) or count ==self.stop:
                update = False
            count+=1

        self.centroids = centroids

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X =x_train

    K = 10
    epochs = 100
    model = Kmeans(X, K, epochs)
    model.fit()
    print(model.centroids)

