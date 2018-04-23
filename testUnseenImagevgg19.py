
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model, Sequential
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load the input image via Keras helper utility
# ensure that the image is resized to 150x150 pixels (expected
# network input dimension of model). Convert the image to a NumPy array
img_path = '/home/ubuntu/data/pancakes_2.jpg'
img = image.load_img(img_path, target_size=(200, 200))
image = image.img_to_array(img)
# Load a compiled model
model = Sequential()
trained_model = load_model('fine_tuned_vgg19_1.h5')
model.add(trained_model)
model.summary()

# image = NumPy array of shape (200, 200, 3),
# need to expand the dimensions to be (1, 200, 200, 3) to
# pass it through the network
image = np.expand_dims(image, axis=0)
print(image.shape)
# assign names to class labels
le = preprocessing.LabelEncoder()
le.fit(["omelette", "hamburger", "chicken_curry", "pancakes", "spaghetti_bolognese", "waffles"])
print(le.classes_)
# classify the image
print("classifying image...")
image_class = model.predict_classes(image)
# Copy an element of an array to a standard Python scalar and return it.
print('The image being tested: ' + img_path)
image_class = image_class.item(0)
print(image_class)
# assign label
class_label = le.inverse_transform([image_class])
print(class_label)
