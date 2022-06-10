import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from Kmeans import Kmeans
from PCA import Pca
from AutoEncoder import AutoEncoder

from IPython.display import display

def Get_Dataset():

    data_dir = 'archive/simpsons_dataset'
    test_dir = 'archive/kaggle_simpson_testset/kaggle_simpson_testset'

    Name=[]
    for file in os.listdir(data_dir):
        if(file !='.DS_Store'):
            Name+=[file]
    print("la liste des personnages/catégories",Name)
    print("nombre de catégories",len(Name))

    N = []
    for i in range(len(Name)):
        N += [i]

    normal_mapping = dict(zip(Name, N))


    def mapper(value):
        return normal_mapping[value]

    dataset=[]
    labels =[]
    count=0
    for name in Name:
        path=os.path.join(data_dir,name)
        for im in os.listdir(path):
            if im[-4:]=='.jpg':
                image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
                image=img_to_array(image)
                image=image/255.0
                dataset.append([image,str(path).split("/")[-1]])
                #labels.append(str(path).split("/")[-1]) #On récupère les noms de chaque personnage sur l'image
        count=count+1

    testset=[]
    for im in os.listdir(test_dir):
        if im[-4:]=='.jpg':
            image=load_img(os.path.join(test_dir,im), grayscale=False, color_mode='rgb', target_size=(100,100))
            image=img_to_array(image)
            image=image/255.0
            testset.append([image,im[0:-4]])

    data,labels=zip(*dataset)
    test,tlabels0=zip(*testset)

    data=np.array(data)
    #labels1=to_categorical(labels0)
    #labels=np.array(labels1)

    test=np.array(test)
    #tlabels=np.array(tlabels0)

    #print(len(data))
    #print(len(test))

    trainx,testx,trainy,testy=train_test_split(data,labels,test_size=0.2,random_state=44)

    #print(trainx.shape)
    #print(testx.shape)
    #print(trainy.shape)
    #print(testy.shape)
    return train_test_split(data,labels,test_size=0.2,random_state=44),Name

def Test_Kmeans(X,Y,name,K,epochs, ligne,col):
    #print(y_train)
    #les différents paramètres testé pour l'entrainement
    #K =50 #(ligne=5, col=10)
    #K =3 #(ligne=1, col=3)
    #K =9 #(ligne=3, col=3)
    #K =14 #(ligne=3, col=5)

    model = Kmeans(X, K, epochs)
    model.fit()


    fig, axs = plt.subplots(ligne, col, figsize=(50, 50))
    plt.gray()

    for i, ax in enumerate(axs.flat):
        if (i < len(model.centroids)):
            ax.imshow(model.centroids[i])
    plt.suptitle("Représentants générés")
    plt.show()

    ## Cette partie ne fonctionne que si le Y_train contient les labels
    cluster = model.cluster
    liste = []
    liste_dict = []
    for k in range(len(model.centroids)):
        liste = np.where(cluster == k)[0]  # liste des indices qui ont k pour centroide le plus proche
        dict_heights = {"maggie_simpson": 0, "simpsons_dataset": 0, "charles_montgomery_burns": 0, "patty_bouvier": 0, "ralph_wiggum": 0, "chief_wiggum": 0, "milhouse_van_houten": 0, "rainier_wolfcastle": 0, "cletus_spuckler": 0, "martin_prince": 0, "lenny_leonard": 0, "sideshow_bob": 0, "fat_tony": 0, "selma_bouvier": 0, "barney_gumble": 0, "lionel_hutz": 0, "gil": 0, "moe_szyslak": 0, "carl_carlson": 0, "edna_krabappel": 0, "snake_jailbird": 0, "groundskeeper_willie": 0, "sideshow_mel": 0, "ned_flanders": 0, "abraham_grampa_simpson": 0, "krusty_the_clown": 0, "waylon_smithers": 0, "apu_nahasapeemapetilon": 0, "marge_simpson": 0, "comic_book_guy": 0, "nelson_muntz": 0, "mayor_quimby": 0, "kent_brockman": 0, "professor_john_frink": 0, "principal_skinner": 0, "bart_simpson": 0, "lisa_simpson": 0, "otto_mann": 0, "troy_mcclure": 0, "miss_hoover": 0, "disco_stu": 0, "homer_simpson": 0, "agnes_skinner": 0}
    
        for x in liste:
            dict_heights[str(Y[x])] = dict_heights[str(Y[x])] + 1
        liste_dict.append(dict_heights)
    
    best=[]
    max_ =0
    #On récupère le pourcentage de représentation de la catégorie max dans chaque cluster
    for repres in liste_dict:
        max_ =max(list(repres.values())) #le nombre de représentation de la catégorie max
        max_perso = list(repres.keys())[list(repres.values()).index(max_)] #le nom de la catégorie max
        pourcentage = max_/sum(list(repres.values()))*100 # le pourcentage de répartition dans ce cluster
    
        best.append((max_perso,pourcentage))
    
    print(best)
    df = pd.DataFrame(np.array(best), columns=['personnage', 'pourcentage'])
    display(df)


def Test_PCA(X,Y,n_components):
    X =np.reshape(
        X, (np.shape(X)[0], np.shape(X)[1] * np.shape(X)[2])
    )
    X = X / 255.0
    model = Pca(X=X, n_components=n_components)
    res = model.fit()

    x = res[:, 0]
    y = res[:, 1]
    # z = res[:,2]
    colors = ['black', 'red', 'green', 'blue', 'cyan', "brown", "gray", "yellow", "forestgreen", "turquoise"]

    fig, ax = plt.subplots()
    # ax = fig.add_subplot(projection='3d')
    c = []
    for i in range(10):
        c.append(ax.scatter(0, 0, color=colors[i], s=0.1))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    colors_scatter = []
    for i in Y:
        colors_scatter.append(colors[i])

    # ax.scatter(x, y, z, c = colors_scatter, s=0.1)
    ax.scatter(x, y, c=colors_scatter, s=0.1)
    ax.legend(handles=c, labels=[str(i) for i in range(10)], loc='center left', bbox_to_anchor=(1, 0.5), markerscale=20)

def Test_AutoEncoder():


    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = x_train
    layers = [28, 16, 8]
    model = AutoEncoder(layers, X,x_test, 'relu',12)
    model.fit(20, 0.01,256)


    encoded = model.encoder(X[0].to(model.dev))
    decoded = model.decoder(encoded.to(model.dev))
    plt.imshow(np.reshape(X[0].numpy(), (25,25, 3)))
    plt.show()




if __name__ == "__main__":

    (x_train, x_test, y_train, y_test),name = Get_Dataset()
    X = x_train
    Y = y_train

    K = 15
    epochs = 200
    ligne = 3
    col = 5
    #Test_Kmeans(X,Y,name,K, epochs, ligne, col)
    n_components = 6
    #Test_PCA(X,Y,n_components)
    Test_AutoEncoder()
