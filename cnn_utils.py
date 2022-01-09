#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 00:48:05 2022

@author: dii
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#Dataset chest_xray
root='./chest_xray'
#print(os.listdir(root))

#pour pointer ver les dossier train test val
train=os.path.join(root,'train')
test=train=os.path.join(root,'test') 
val =train=os.path.join(root, 'val') 

#print(os.listdir(train))

#lecture des images avec ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255.)
#instanciation de notre objet crée

