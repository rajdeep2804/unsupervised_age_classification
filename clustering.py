from unsupervised_age_classification import KMeans

path = (r'c:\Users\hp\pictures\Food\cluster')
for i in range(0,3):
    p = os.path.sep.join([path,str(i)])
    if not os.path.exists(path):
        os.makedirs(p)
        
image_test_data = pd.read_csv(r'C:\Users\hp\Pictures\Food\output_dir\feature_vec_test.csv')
path_0 = (r'c:\Users\hp\pictures\Food\cluster\0')
path_1 = (r'c:\Users\hp\pictures\Food\cluster\1')
path_2 = (r'c:\Users\hp\pictures\Food\cluster\2')
f = (r'c:\users\hp\pictures\dataset_unsupervised_test')
image_test_name = image_test_data.columns[0]
image_test_features = image_test_data.drop([image_test_name], axis = 1)
image_test_name = np.array(image_test_name)
pred_clusters = KMeans.kmeans_model.predict(image_test_features)

d = os.path.join(f, image_test_labels)
image_paths = list(paths.list_images(d))
for image_path in image_paths:

    if pred_clusters == str(0):
        shutil.copy(image_path, path_0)
    elif pred_clusters == str(1):
        shutil.copy(image_path,path_1)
    elif pred_clusters == str(2):
        shutil.copy(image_path, path_2)

