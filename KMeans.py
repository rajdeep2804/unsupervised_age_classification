import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from imutils import paths
import os
import shutil

path = (r'c:\Users\hp\pictures\Food\cluster')
for i in range(0,3):
    p = os.path.sep.join([path,str(i)])
    if not os.path.exists(path):
        os.makedirs(p)

image_data = pd.read_csv(r'C:\Users\hp\Pictures\Food\feature_vec.csv')
image_test_data = pd.read_csv(r'C:\Users\hp\Pictures\Food\output_dir\feature_vec_test.csv')
image_paths = path = r'C:\Users\hp\Pictures\dataset_unsupervised_test'
print(image_data.shape)
print(image_test_data.shape)
path_0 = (r'c:\Users\hp\pictures\Food\cluster\0')
path_1 = (r'c:\Users\hp\pictures\Food\cluster\1')
path_2 = (r'c:\Users\hp\pictures\Food\cluster\2')
f = (r'c:\users\hp\pictures\dataset_unsupervised_test')

kmeans_model = KMeans(n_clusters=3, max_iter=300).fit(image_data)
kmeans_centroids = kmeans_model.cluster_centers_
np.unique(kmeans_model.labels_)
image_test_name = image_test_data.columns[0]
image_test_features = image_test_data.drop([image_test_name], axis = 1)
image_test_name = np.array(image_test_name)
pred_clusters = kmeans_model.predict(image_test_features)

d = os.path.join(f, image_test_labels)
image_paths = list(paths.list_images(d))
for image_path in image_paths:

    if pred_clusters == str(0):
        shutil.copy(image_path, path_0)
        print("pass")
    elif pred_clusters == str(1):
        shutil.copy(image_path,path_1)
        print("pass1")
    elif pred_clusters == str(2):
        shutil.copy(image_path, path_2)

