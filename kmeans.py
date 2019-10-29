import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.datasets import load_sample_image
from sklearn.cluster import KMeans

def reconstruct_image(cluster_centers, labels, w, h):
    d = cluster_centers.shape[1]
    image = np.zeros((w, h, d))
    label_index = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = cluster_centers[labels[label_index]]
            label_index += 1
    return image

flower = load_sample_image('flower.jpg')
flower = np.array(flower, dtype=np.float64) / 255
plt.imshow(flower)
#plt.show()

print(flower.shape) # (427,640,3)
w, h, d = original_shape = tuple(flower.shape)
#print(w,h,d) # 427, 640, 3
#assert d == 3
image_array = np.reshape(flower, (w * h, d))
print(image_array.shape) # (273280,3)

# Setting  Kmeans
image_sample = shuffle(image_array, random_state=42)[:1000]
n_colors = 10
kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(image_sample)
#Get color indices for full image
labels = kmeans.predict(image_array)

# Plots
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image with 96 615 colors')
plt.imshow(flower)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Reconstructed image with 64 colors')
plt.imshow(reconstruct_image(kmeans.cluster_centers_, labels, w, h))
plt.show()

