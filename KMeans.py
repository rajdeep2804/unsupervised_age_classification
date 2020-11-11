import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from imutils import paths
import os
import shutil

image_data = pd.read_csv(r'C:\Users\hp\Pictures\Food\feature_vec.csv')
print(image_data.shape)
print(image_test_data.shape)


kmeans_model = KMeans(n_clusters=3, max_iter=300).fit(image_data)
kmeans_centroids = kmeans_model.cluster_centers_
np.unique(kmeans_model.labels_)

