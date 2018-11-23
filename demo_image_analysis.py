# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:31:58 2018

@author: Zhangyu Li, Xiaofei Zhu 
zxf@cqut.edu.cn
"""
import os
import numpy as np
from keras.utils.np_utils import to_categorical

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

def load_data():
    images = []
    labels = []
    test_images = []
    for f in os.listdir('img_path_category0'):
        img_path = os.path.join('矩形', f)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        images.append(x)
        labels.append(0)
        
    for f in os.listdir('img_path_category1'):
        img_path = os.path.join('img_path_category1', f)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        images.append(x)
        labels.append(1)
    
    for f in os.listdir('img_path_category2'):
        img_path = os.path.join('img_path_category2', f)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        images.append(x)
        labels.append(2)
	
    for f in os.listdir('test_path'):
        img_path = os.path.join('test_path', f)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        test_images.append(x)
	
    images = np.array(images)
    test_images = np.array(test_images)
    return images, labels, test_images

images, labels, test_images = load_data()
labels = to_categorical(labels)

#base_model = VGG16(include_top=False)
base_model = VGG16(weights='imagenet', include_top=False)

# add pool_layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100, activation='relu')(x)
h = Dense(3, activation='softmax')(x)
 
model = Model(inputs=base_model.input, outputs=h)


for layer in base_model.layers:
    layer.trainable = False

# compile model (only after block aforementioned layers)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(images, labels, batch_size=6, epochs=2)


# get layer names
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we block the bottom 15 layers and only fine-tune the remaining layers of VGG16 
for layer in model.layers[:15]:
   layer.trainable = False
for layer in model.layers[15:]:
   layer.trainable = True

# we need to recompile the model
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit(images, labels, batch_size=6, epochs=6)



