from Kmeans import Kmeans
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np

####KMeans
#Tache de Generation: Genère des représentants
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = x_train

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
representant =[]
heights = []
for k in range(len(k_means.centroids)):
    representant.append(k)
    heights.append(np.sum(np.where(cluster == k)))

plt.bar(representant,heights)
plt.title("Repartition du dataset à chaqsue représentant")
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




