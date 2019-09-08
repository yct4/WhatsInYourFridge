from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np

from PIL import Image
import os

# Training data directory
train_data_dir = 'train'

# Test data
validation_data_dir = 'test'

# Total samples (below x 2 for good and bad)
nb_train_samples = 96
nb_validation_samples = 55

img_width = 300
img_height = 300

# Define epoch and batch size (what exactly is this?)
epochs = 10
batch_size = 10

# Check to ensure images are correct size (how images are formatted)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape=input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(2)) 
model.add(Activation('sigmoid'))


model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy']) 

print("Model compiled\n")

# Load and label data
train_datagen = ImageDataGenerator( 
    rescale=1. / 255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True) 
print("Loaded data\n")

test_datagen = ImageDataGenerator(rescale=1. / 255) 
  
train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='categorical', 
    classes=['fruit', 'vegetable'])

print("train generated\n")
print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory( 
    validation_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='categorical', 
    classes=['fruit', 'vegetable']) 
  
print("test generated\n")
steps_per_epoch = nb_train_samples // batch_size
validation_steps = nb_validation_samples // batch_size
model.fit_generator( 
    train_generator, 
    steps_per_epoch = steps_per_epoch, 
    epochs = epochs, 
    validation_data = validation_generator, 
    validation_steps = validation_steps) 

print("Model fit\n")
model.save_weights('food_classifier.h5') 

test_image_dir_path = 'test/fruit/'
test_image = []
for file in os.listdir(test_image_dir_path): 
  img = image.load_img(test_image_dir_path + file, target_size= (img_width, img_height))
  img = image.img_to_array(img)
  img = img/255
  test_image.append(img)

test_image_dir_path_veg = 'test/vegetable/'
test_image_veg = []
for file in os.listdir(test_image_dir_path_veg): 
  img = image.load_img(test_image_dir_path_veg + file, target_size= (img_width, img_height))
  img = image.img_to_array(img)
  img = img/255
  test_image_veg.append(img)

test = np.array(test_image)
test_veg = np.array(test_image_veg)

print("Fruit Test \n")
prediction = model.predict_classes(test)
print("predictions: ")
idx = 0 ;
for i in prediction : 
    if ( i == 0 ) :
        print (str(idx) + ':fruit, ')
    else :
        print (str(idx) + ':vegetable, ')
    idx = idx + 1
print('\n')

print("Veggie Test\n")
prediction = model.predict_classes(test_veg)
print("predictions: ")
idx = 0 ;
for i in prediction : 
    if ( i == 0 ) :
        print (str(idx) + ':fruit, ')
    else :
        print (str(idx) + ':vegetable, ')
    idx = idx + 1


#print(prediction)