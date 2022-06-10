import numpy as np
import random

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


class Kmeans:
    X: np.ndarray  # chaque ligne est une image
    # Y: int
    K: int
    stop: int
    cluster = []
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
        self.cluster = centroids_label
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

    def encode(self, point):
        """Cette fonction est utilisée pour encoder un point,
        en prenant un noouveau point quel output on aura en fonction des distances des représentants"""
        distances = np.empty(len(self.centroids))
        for j, c in enumerate(self.centroids):
            distances[j] = np.linalg.norm(point - c)

        return self.centroids[np.argmin(distances)]



if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X =x_train
    Y=y_train
    K = 14
    epochs = 200
    model = Kmeans(X, K, epochs)
    model.fit()


    colors = 10 * ["r", "g", "c", "b", "k"]

    fig, axs = plt.subplots(3, 5, figsize=(50, 50))
    plt.gray()

    for i, ax in enumerate(axs.flat):
        if(i <len(model.centroids)):
            ax.imshow(model.centroids[i])
    plt.suptitle("Représentants générés")
    plt.show()


    cluster = model.cluster
    liste = []
    liste_dict = []
    for k in range(len(model.centroids)):
        liste = np.where(cluster == k)[0]  # liste des indices qui ont k pour centroide le plus proche
        dict_heights = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}

        for x in liste:
            dict_heights[str(Y[x])] = dict_heights[str(Y[x])] + 1
        liste_dict.append(dict_heights)

    col = 5
    images = []
    repres = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for x in range(model.K):
        heights = list(liste_dict[x].values())
        plt.subplot(int(len(repres) / col + 1), col, x + 1)
        plt.bar(repres, heights)

    plt.suptitle("Repartition du dataset pour chaque représentant")
    plt.show()

