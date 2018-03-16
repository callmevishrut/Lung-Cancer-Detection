# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:26:33 2018

@author: Vishrut Sharma
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
classifier = load_model('dog_cat_cnn.h5')
test_image = image.load_img('dataset/single_prediction/dog_or_cat.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print (prediction)