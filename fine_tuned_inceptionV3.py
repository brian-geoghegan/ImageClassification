
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
from keras.applications import InceptionV3
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import os
import time

modelCreationStart = time.time()//60

import matplotlib.pyplot as plt

base_dir = '/home/ubuntu/data/food101_selected/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
# Instantiating a small CNN

conv_base = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(200, 200, 3))

# Freeze the layers except the last 26 layers
for layer in conv_base.layers[:-26]:
    layer.trainable = False

print("Layers in model" + str(len(conv_base.layers)))

basemodel = conv_base.output
basemodel = Flatten()(basemodel)
basemodel = Dense(1024, activation='relu')(basemodel)
basemodel = Dropout(0.5)(basemodel)
basemodel = Dense(1024, activation='relu')(basemodel)
predictions = Dense(6, activation='softmax')(basemodel)

model = Model(input = conv_base.input, output = predictions)

learning_rate = 0.01
epochs = 50
decay = learning_rate/epochs

# Configuring our model for training
# For a categorical classification problem
model.compile(loss='categorical_crossentropy',
 optimizer=optimizers.RMSprop(lr=1e-4),
 metrics=['acc'])

# Using ImageDataGenerator to read images from directories
# all images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
shear_range=0.2,
height_shift_range=0.2,
width_shift_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
 train_dir, # this is the target directory
 target_size=(200, 200), # all images will be resized to 150x150
 batch_size=32,
 class_mode='categorical')

 # since we use binary_crossentropy loss, we need binary labels
validation_generator = validation_datagen.flow_from_directory(
 validation_dir,
 target_size=(200, 200),
 batch_size=20,
 class_mode='categorical')

reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01)
checkpoint = ModelCheckpoint("fine_tuned_InceptionV3_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


history = model.fit_generator(
train_generator,
steps_per_epoch=150,
epochs=epochs,
shuffle=True,
validation_data=validation_generator,
validation_steps=50,
verbose=1,
callbacks = [reduceLR, checkpoint, early])

modelCreationEnd = time.time()//60
print("Time taken to build the model: " + str(modelCreationEnd - modelCreationStart))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/ubuntu/data/cnn_inceptionV3.png')


# Evaluating the model - test on finalised model
# Returns the loss value & metrics values for the model in test mode.
test_generator = test_datagen.flow_from_directory(
 test_dir,
 target_size=(200, 200),
 batch_size=20,
 class_mode='categorical')
# finally evaluate this model on the test data
print("Beginning model evaluation...")
modelTestStart = time.time()//60

results = model.evaluate_generator(
 test_generator,
 steps=1000)
modelTestEnd = time.time()//60

print("Time taken to test the model: " + str(modelTestEnd - modelTestStart))

print('Final test accuracy:', (results[1]*100.0))

