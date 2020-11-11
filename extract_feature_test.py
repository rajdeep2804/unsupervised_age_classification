from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import random
import os
path = r'C:\Users\hp\Pictures\dataset_unsupervised_test'
feature_extraction = r'C:\Users\hp\Pictures\Food\output_dir'
feature_vec_test = "feature_vec_test"
batch_size = 32 
model = VGG16(weights="imagenet", include_top=False)
csvPath = os.path.sep.join([feature_extraction, "{}.csv".format(feature_vec_test)])
csv = open(csvPath, 'a')
for image_paths in os.listdir(path):
    input_path = os.path.join(path, image_paths)
    tail = os.path.split(input_path)
    file_name = tail[1]
    image = load_img(input_path, target_size = (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, batch_size=batch_size)
    features = features.reshape((features.shape[0], 7 * 7 * 512))
    for vec in features:
        vec = ",".join([str(v) for v in vec])
        csv.write("{},{}\n".format(file_name,vec))
                
    
csv.close()
