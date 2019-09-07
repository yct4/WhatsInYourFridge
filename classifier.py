from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import Image

img_width = 150
img_height = 70

# Training data directory
train_data_dir = 'training_images'

# Test data
validation_data_dir = 'test_images'

# Total samples (below x 2 for good and bad)
nb_train_samples = 222
nb_validation_samples = 20


# Define epoch and batch size (what exactly is this?)
epochs = 10
batch_size = 16

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
model.add(Dense(1)) 
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy']) 

# Load and label data
train_datagen = ImageDataGenerator( 
    rescale=1. / 255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True) 
  
test_datagen = ImageDataGenerator(rescale=1. / 255) 
  
train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='binary', 
    classes=['bad', 'good']) 

print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory( 
    validation_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='binary', 
    classes=['bad', 'good']) 
  
model.fit_generator( 
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size, 
    epochs=epochs, 
    validation_data=validation_generator, 
    validation_steps=nb_validation_samples // batch_size) 

model.save_weights('teeth_classifier.h5') 

test_image_dir_path = 'C:/Users/nico/teethclassification/test_images/bad/'
test_image = []
img = image.load_img(test_image_dir_path + 'test_' + str(1) + '.jpg', target_size= (img_width, img_height))
img = image.img_to_array(img)
img = img/255
test_image.append(img)
test = np.array(test_image)
prediction = model.predict_classes(test)
print("predictions: ")
print(prediction)