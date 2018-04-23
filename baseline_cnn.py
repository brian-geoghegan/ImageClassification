import numpy
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from scipy.misc import toimage
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import time
import os

base_dir = '/home/ubuntu/data/food101_selected/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# fix dimension ordering issue
from keras import backend as K
K.set_image_dim_ordering('th')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 200, 200), padding='same',
activation='relu', kernel_constraint=maxnorm(3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(6, activation='softmax'))

# Compile model
epochs = 15
lrate = 0.01
decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

train_datagen = ImageDataGenerator(rescale=1./255)
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

start_training_phase = time.time()//60
history = model.fit_generator(
train_generator,
steps_per_epoch=150,
epochs=epochs,
shuffle=True,
validation_data=validation_generator,
validation_steps=50)
# Saving our model
model.save('baseline_cnn_food.h5')
end_training_phase = time.time()//60
print("Time taken to train the model: " + str(end_training_phase - start_training_phase))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/ubuntu/data/cnn_baseline.png')

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

