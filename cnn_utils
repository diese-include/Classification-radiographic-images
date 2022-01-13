#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:17:59 2022

@author: dii
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 00:48:05 2022

@author: dii
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#dossier racine
racine_path = './chest_xray'

#sous dossier du repertoire racine
train_rep_path = racine_path + '/train/'
test_rep_path = racine_path + '/test/'
val_rep_path = racine_path + '/val/'

##dans le dossier train chemin pour accéder aux dossier NORMAL ET PNEUMONIA
train_n_path = train_rep_path+'/NORMAL/'
train_p_path = train_rep_path+'/PNEUMONIA/'

#dans le dossier test chemin pour accéder aux dossier NORMAL ET PNEUMONIA
test_n_path = test_rep_path+'/NORMAL/'
test_p_path = test_rep_path+'/PNEUMONIA/'


"""
#nombre d'images pour l'entrainement poumons normaux
print(len(os.listdir(train_n_path))) 
#nombre d'images pour l'entrainement poumons pneumonie
print(len(os.listdir(train_p_path))) 
print('Total images pour l\'entrainement', len(os.listdir(train_n_path)) + len(os.listdir(train_p_path)))
"""
# Choisissons une image au hasard dans le dossier pneumonia
normal= np.random.randint(0,len(os.listdir(train_n_path))) 
normal_img = os.listdir(train_n_path)[normal]
normal_img_adr = train_n_path+normal_img

# Choisissons une image au hasard dans le dossier pneumonia
pneumonie = np.random.randint(0,len(os.listdir(train_p_path)))
pneumonie =  os.listdir(train_p_path)[pneumonie]
pneumonia_img_adr = train_p_path+pneumonie

#chargement 2 images
Img_1 = Image.open(normal_img_adr)
Img_2 = Image.open(pneumonia_img_adr)


#Visualisation
plt.figure(figsize= (13,13))
plt.subplot(1,2,1)
plt.title('Poumon Normal')
plt.imshow(Img_1, cmap = 'gray')


plt.subplot(1, 2, 2)
plt.title('Pneumonie')
plt.imshow(Img_2, cmap = 'gray')



img_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   validation_split = 0.25,
                                   horizontal_flip = True)


training_set = img_datagen.flow_from_directory('./chest_xray/train',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 color_mode='grayscale',
                                                 subset='training',
                                                 class_mode = 'binary')

validation_generator = img_datagen.flow_from_directory('./chest_xray/train',
                                                        target_size=(256, 256),
                                                        batch_size=32,
                                                        subset="validation",
                                                        color_mode='grayscale',
                                                        class_mode='binary')

test_set = img_datagen.flow_from_directory('./chest_xray/test',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            color_mode='grayscale',
                                            shuffle=False,
                                            class_mode = 'binary')

warnings.filterwarnings('ignore')

nb_classe=2

model=keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', 
                        input_shape=(256,256,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(nb_classe,activation='softmax')     
])


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']    
)

#########################################################################
model.summary()
##################################################################

# entrainement model
"""
history=model.fit(
    training_set,
    validation_data=validation_generator,
    epochs=1,
)
"""

history = model.fit(training_set,
                      epochs = 1 ,
                      validation_data = validation_generator,
                      )

###############################################################################
#Évaluer la précision(Nous allons comparer les performances du modèle sur l'ensemble de données de test)

test_loss, test_acc = model.evaluate(test_set,verbose=2)
print('\nTest accuracy:', test_acc)

print(test_loss)
################################################################################


#######################################################################
#Faire des prédictions
#predictions = model.predict(test_set)
#predictions = (predictions>0.5).astype(np.int)

predictions =model.predict(test_set)

#Ici, le modèle a prédit l'étiquette pour chaque image dans l'ensemble de test.
# Jetons un coup d'œil à la première prédiction :
predictions[0]
print(predictions[0])

#print('classe 1 normal',np.argmax(predictions[0]))

############################################################################
#class_names = ['Normal', 'Pneumonia']
#print(class_names[0])

###############################################


  












