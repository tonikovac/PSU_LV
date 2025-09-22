from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def generate_data(n_samples, flagc):
    
    if flagc == 1:
        random_state = 365
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
        
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X


#1 zad
data = generate_data(500, 1)
print(type(data))
print(data.shape)
kmeans = KMeans(random_state=42).fit(data)
print(kmeans.labels_)
kmeans.predict(data)
print(kmeans.cluster_centers_)

#2 zad
inertias =[]
cluster_range = range(1,21)
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertias, marker='o')
plt.xlabel('Broj klastera')
plt.ylabel('Vrijednost kriterijske funkcije (Inertia)')
plt.title('Elbow metoda za određivanje optimalnog broja klastera')
plt.grid(True)
plt.savefig('elbow_metoda.png')

#zad 3
# Hijerarhijsko grupiranje
linked = linkage(data, method='single')

# Crtanje dendograma
plt.figure(figsize=(12, 6))
dendrogram(linked)
plt.title('Dendrogram - metoda: single')
plt.xlabel('Primjeri')
plt.ylabel('Udaljenost')
plt.savefig('dendogram.png')

methods = ['single', 'complete', 'average', 'ward']

for method in methods:
    linked = linkage(data, method=method)
    
    plt.figure(figsize=(12, 6))
    dendrogram(linked)
    plt.title(f'Dendrogram - metoda: {method}')
    plt.xlabel('Primjeri')
    plt.ylabel('Udaljenost')
    plt.savefig(f'dendogram_{method}.png')



