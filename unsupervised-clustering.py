import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
import os
import time
import random

# Reference: https://github.com/beleidy/unsupervised-image-clustering

class data(object):

    def __init__(self):
        self.IMAGE_FOLDER = './inputs/'        # Images folder
        self.IMAGE_TYPE='.png'      # Define Image type

    def load_data(self):
        # This function loads
        # 1) the images contained inside the folder '../all_images/images/'
        # 2) the mask (or label) images contained inside the folder '../all_images/images_label/'

        # Get all images name inside folder (IMAGE_FOLDER)
        images = [x for x in sorted(os.listdir(self.IMAGE_FOLDER)) if x[-4:] == self.IMAGE_TYPE]

        # Sort Images
        x_data = [] # np.empty((len(all_images), self.IMG_HEIGHT, self.IMG_WIDTH), dtype='float32')
        y_data = []

        for name in images:
          
            im = cv2.imread(self.IMAGE_FOLDER + name, 0)
            im = cv2.resize(im, (224, 224))
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) # Covert image to color
            im = im/255 # normalize
            x_data.append(im) # Vector containing all the images in range 0 255 (they are not normalized)
            y_data.append(name)

        np.save("inputs_x_data.npy", x_data)
        np.save("inputs_y_data.npy", y_data)
        
        return x_data, y_data

def covnet_transform(covnet_model, raw_images):
    pred = covnet_model.predict(raw_images)     # Pass our training data through the network
    flat = pred.reshape(raw_images.shape[0], -1)     # Flatten the array
    return flat

def create_fit_PCA(data, n_components=None):
    # n_components == min(n_samples, n_features) - 1
    p = PCA(n_components=n_components, random_state=728)
    p.fit(data)
    return p

def pca_cumsum_plot(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

def create_train_kmeans(data, number_of_clusters):
    # n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    # especially when the data size gets much bigger. #perfMatters

    k = KMeans(n_clusters=number_of_clusters, n_jobs=-1, random_state=728)

    # Let's do some timings to see how long it takes to train.
    start = time.time()

    # Train it up
    k.fit(data)

    # Stop the timing
    end = time.time()

    # And see how long that took
    print("Training took {} seconds".format(end-start))

    return k

def write_results(x_data, y_data, k_vgg16_pred_pca, number_of_clusters):
    for i in range(0,  number_of_clusters):
        name = "results/"+ str(i)
        if not os.path.exists(name):
            os.mkdir(name)
    for i in range(0, x_data.shape[0]):
        name = "results/" + str(k_vgg16_pred_pca[i]) + "/" + y_data[i]
        cv2.imwrite(name,255*x_data[i,:,:,:])

def create_dataset(number_of_clusters):
    for i in range (0, number_of_clusters):
        name = "results/"+ str(i)
        if not os.path.exists(name):
            os.mkdir(name)
        images = [x for x in sorted(os.listdir(name)) if x[-4:] == ".png"]
        list_of_random_items = random.sample(images, 12)

        for filename in list_of_random_items:
            im = cv2.imread("inputs" + "/" + filename, 0)
            cv2.imwrite("results/dataset/" + filename,im)

def elbow_method(vgg16_output_pca):
    wcss = []
    for i in range(1,15):
        kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
        kmeans.fit(vgg16_output_pca)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,15),wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('elbow.png')
    plt.show()

def elbow_method_2(vgg16_output_pca):
    from scipy.spatial.distance import cdist
    # k means determine k
    distortions = []
    K = range(1,25)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(vgg16_output_pca)
        kmeanModel.fit(vgg16_output_pca)
        distortions.append(sum(np.min(cdist(vgg16_output_pca, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / vgg16_output_pca.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


Data = data()
#x_train = Data.load_data()
x_data= np.load("inputs_x_data.npy")
y_data= np.load("inputs_y_data.npy")

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.001, random_state=42)

vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
#vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
#resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
vgg16_output = covnet_transform(vgg16_model, x_train)
print("VGG16 flattened output has {} features".format(vgg16_output.shape[1]))
vgg16_pca = create_fit_PCA(vgg16_output )
#pca_cumsum_plot(vgg16_pca)
vgg16_output_pca = vgg16_pca.transform(vgg16_output)

elbow_method_2(vgg16_output_pca)
number_of_clusters=15

K_vgg16_pca = create_train_kmeans(vgg16_output_pca,number_of_clusters)
#K_vgg16 = create_train_kmeans(vgg16_output,number_of_clusters)

# KMeans with PCA outputs
k_vgg16_pred_pca = K_vgg16_pca.predict(vgg16_output_pca)
print(k_vgg16_pred_pca)
print(k_vgg16_pred_pca.shape)

# KMeans with CovNet outputs
#k_vgg16_pred = K_vgg16.predict(vgg16_output)

write_results(x_train, y_train, k_vgg16_pred_pca, number_of_clusters)

#vgg16_cluster_count = cluster_label_count(k_vgg16_pred, y_train)
#vgg16_cluster_count_pca = cluster_label_count(k_vgg16_pred_pca, y_train)

number_of_clusters=15
create_dataset(number_of_clusters)
