from Kmeans import Kmeans
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np

####KMeans
#Tache de Generation: Genère des représentants
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = x_train
Y = y_train

K = 9
epochs = 200
k_means = Kmeans(X, K, epochs)
k_means.fit()
#print(k_means.centroids)


# Plotting starts here
colors = 10 * ["r", "g", "c", "b", "k"]

fig, axs = plt.subplots(3, 3, figsize = (12, 12))
plt.gray()

for i, ax in enumerate(axs.flat):
    ax.imshow(k_means.centroids[i])
plt.title("Représentants générés")
plt.show()

#Visualisation : renvoie la repartition des points selon leur representant le plus proche
cluster = k_means.cluster
liste = []
liste_dict =[]
for k in range(len(k_means.centroids)):
    liste = np.where(cluster == k)[0] #liste des indices qui ont k pour centroide le plus proche
    dict_heights = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}

    for x in liste:
        dict_heights[str(Y[x])]=dict_heights[str(Y[x])]+1
    liste_dict.append(dict_heights)

col = 3
images =[]
repres = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
for x in range(k_means.K):
    heights = list(liste_dict[x].values())
    plt.subplot(int(len(repres) / col+1), col, x + 1)
    plt.bar(repres,heights)

plt.suptitle("Repartition du dataset pour chaque représentant")
plt.show()

#Compression/Decompression : renvoyer ce qui se trouve dans l'espace latent pour avoir estimer à quel point on peut compresser une image
encoded = k_means.encode(X[6])
images = [np.reshape(encoded, (28, 28)), np.reshape(X[6], (28, 28))]
plt.figure(figsize=(12,12))
columns = 2
for i, image in enumerate(images):
    plt.subplot(int(len(images) / columns + 1), columns, i + 1)
    plt.imshow(image, cmap='gray')
plt.show()




