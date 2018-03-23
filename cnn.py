import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

batch_size = 10
num_classes = 2
batches = 10
img_width, img_height = 300, 300

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_samples = 381
epochs = train_samples / batch_size

train = 'train'

train_batch = ImageDataGenerator().flow_from_directory(train ,
                                                     target_size = (300,300) ,
                                                     classes = ['jump','nothing'],
                                                     batch_size = 10)

img, labels = next(train_batch)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale = 1. /255,
    horizontal_flip=True)

model.fit_generator(train_batch,
                    steps_per_epoch=10,
                    epochs=epochs)

model.save_weights('my_model_weights.h5')
model.save('my_model.h5')