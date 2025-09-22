import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image = mpimg.imread('example_grayscale.png')

print(image.shape)

'''plt.figure(figsize=(6,6))
plt.imshow(image, cmap='gray')
plt.title('Originalna slika')
plt.axis('off')
plt.show()'''

X = image.reshape(-1, 1)

n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)

values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_

image_compressed = np.choose(labels, values)
image_compressed = image_compressed.reshape(image.shape)

plt.figure(figsize=(6,6))
plt.imshow(image_compressed, cmap='gray')
plt.title(f'Kvantizirana slika ({n_clusters} klastera)')
plt.axis('off')
plt.savefig('kvantizirana_slika.png')

original_bits_per_pixel = 8 
compressed_bits_per_pixel = np.ceil(np.log2(n_clusters))

compression_ratio = compressed_bits_per_pixel / original_bits_per_pixel

print(f'Kompresijski omjer: {compression_ratio:.2f}')
print(f'Smanjenje veličine: {(1 - compression_ratio) * 100:.2f}%')

#5 zad
image = mpimg.imread('example.png')

print(image.shape)

X = image.reshape(-1, 3)

print(X.shape)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

image_quantized = cluster_centers[labels]
image_quantized = image_quantized.reshape(image.shape)

plt.figure(figsize=(8,8))
plt.imshow(image_quantized)
plt.title(f'Kvantizirana slika ({n_clusters} klastera)')
plt.axis('off')
plt.savefig('kvantizirana_slika_5zad.png')


