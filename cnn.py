# -*- coding: utf-8 -*-


# Importing the keras libraries and packages
import h5py
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Initializing CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3),
                      activation = 'relu'))
"""
the first is the number of filters i.e 32 
here, the second argument is the shape each filter is going to be 
i.e 3x3 here, the third is the input shape and 
the type of image(RGB or Black and White)of each image i.e 
the input image our CNN is going to be taking is of a 64x64 resolution 
and “3” stands for RGB, which is a colour img, 
the fourth argument is the activation function we want to use,
 here ‘relu’ stands for a rectifier function.
"""

"""
The primary aim of a pooling operation is to reduce the size of the images 
as much as possible. In order to understand what happens in these steps in
 more detail you need to read few external resources.
 But the key thing to understand here is that 
we are trying to reduce the total number of nodes for the upcoming layers.
"""

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
"""
It’s time for us to now convert all the pooled images into a continuous
vector through Flattening. 
Flattening is a very important step to understand. 
What we are basically doing here is taking the 2-D array, 
i.e pooled image pixels and
converting them to a one dimensional single vector.
"""
classifier.add(Flatten())


classifier.add(Dense(units = 128, activation = 'relu'))

"""
As you can see, Dense is the function to add a fully connected layer,
‘units’ is where we define the number of nodes that should be present
in this hidden layer, these units value will be always between the number 
of input nodes and the output nodes but the art of choosing the most
 optimal
 number of nodes can be achieved only through experimental tries. 
 Though it’s a common practice to use a power of 2.
And the activation function will be a rectifier function.
"""

classifier.add(Dense(units = 1, activation = 'sigmoid'))
#compliing the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# now we need to fit the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
history = classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 64)

#to plot a graph for loss and accuracy on training and testing
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#saving the modal
classifier.save('dog_cat_cnn.h5')

#Now we have to make predictions




