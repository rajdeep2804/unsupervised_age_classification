# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import random
import os
path = r'C:\Users\hp\Pictures\dataset_unsupervised'
feature_extraction = r'C:\Users\hp\Pictures\Food\output_dir'
feature_vec = "feature_vec"
dataset_unsupervised = "dataset_unsupervised"
batch_size = 32 
model = VGG16(weights="imagenet", include_top=False)
csvPath = os.path.sep.join([feature_extraction, "{}.csv".format(feature_vec)])
print(csvPath)                            
for image_paths in os.listdir(path):
    input_path = os.path.join(path, image_paths)
    print("input_path",input_path)
    csv = open(csvPath, 'a')
    image = load_img(input_path, target_size = (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, batch_size=batch_size)
    features = features.reshape((features.shape[0], 7 * 7 * 512))
    for vec in features:
        vec = vec = ",".join([str(v) for v in vec])
        csv.write("{}\n".format(vec))
                
    
csv.close()
