
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


#lire l'ensemble de donnée avec  ImageDataGenerator on crée l'objet puis on l'instancie 
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

#nombre de classe
nb_classe=2


#modèle AlexNet
model=keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', 
                        input_shape=(256,256,1)), #fichier d'entrée taille 256x256
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),             #pas par defaut strides=(1,1)
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


#compilation du modèle
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']    
)

#########################################################################
model.summary()
##################################################################

# entrainement model

history = model.fit(training_set,
                      epochs = 30 ,
                      validation_data = validation_generator,
                      )

###############################################################################
#Évaluer la précision(Nous allons comparer les performances du modèle sur l'ensemble de données de test)

test_loss, test_acc = model.evaluate(test_set,verbose=2)
print('\nTest accuracy:', test_acc)

#taux de perte
print('Perte',test_loss)
################################################################################


#######################################################################
#Faire des prédictions sur l'ensemble des données tests:(on obtient un array
#contenant des array contenant 2 elts chacun pour chaque images,ces valeurs sont des proba. 2 elts parcequ'il y a juste 2 classes
#l'image appartient à la classe dont la proba est élevé)
#ex:si la prédiction de l'image 5 renvoie ceci [0.6,0.7] on dira
#que l'image appartient à la classe pneumonie [NORMAL,PNEUMONIA]

predictions =model.predict(test_set)
#print(predictions)

# Jetons un coup d'œil à la première prédiction :
predictions[1]
print('Premiere prediction:',predictions[1])
#pour choisir la valeur maximale du tableau

#attention à la manière dont python compte 0,1,2,... et non 1,2,3,....
print('Classe:',np.argmax(predictions[1]))

#classe 0:NORMAL
#classe 1:PNEUMONIA

############################################################################


###############################################


  












