from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split  # To split dataset into training and validation
import os

# Reference: http://dilloncamp.com/projects/pca.html

#fit(self, X[, y])	Fit the model with X.
#fit_transform(self, X[, y])	Fit the model with X and apply the dimensionality reduction on X.
#get_covariance(self)	Compute data covariance with the generative model.
#get_params(self[, deep])	Get parameters for this estimator.
#get_precision(self)	Compute data precision matrix with the generative model.
#inverse_transform(self, X)	Transform data back to its original space.
#score(self, X[, y])	Return the average log-likelihood of all samples.
#score_samples(self, X)	Return the log-likelihood of each sample.
#set_params(self, \*\*params)	Set the parameters of this estimator.
#transform(self, X)	Apply dimensionality reduction to X.


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

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

        return x_train, x_test, y_train, y_test

def plot(X_norm, approximation):
    for i in range(0,X_norm.shape[0]):
        #X_norm[i,] = X_norm[i,].T
        #approximation[i,] = approximation[i,].T
        cv2.imshow("Original Image", 255*X_norm[i,])
        cv2.imshow("Reconstructed Image", 255*approximation[i,])
        cv2.waitKey(0)

def do_pca(x_train, x_test):
    # x_train is a vector o gray scale images. [number_of_images, width, height]
    # x_test is a vector o gray scale images. [number_of_images, width, height], this images are used to test if the pca developed model is   correct
    #Normalize data by subtracting mean and scaling
    X_norm = normalize(x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))

    #Set pca to find principal components that explain 99%
    #of the variation in the data
    pca = PCA(.9)
    #Run PCA on normalized image data
    lower_dimension_data = pca.fit_transform(X_norm)
    #Lower dimension data is 5000x353 instead of 5000x1024
    print(lower_dimension_data.shape)


    #Project lower dimension data onto original features
    approximation = pca.inverse_transform(lower_dimension_data)
    #Approximation is 5000x1024
    print(approximation.shape)
    #Reshape approximation and X_norm to 5000x32x32 to display images
    approximation = approximation.reshape(x_train.shape)
    X_norm = X_norm.reshape(x_train.shape)

    plot(X_norm, approximation)


Data = data()
x_train, x_test, y_train, y_test = Data.load_data()
do_pca(x_train, x_test)
