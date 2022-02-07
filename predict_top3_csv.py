#!/usr/bin/python
# This script prints the top 3 predicted and real body parts in the validation dataset (images/test)

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.preprocessing import image
import csv
import re

#csv
csvFile = open('predict_top3.csv', 'a', newline="")
csvWriter = csv.writer(csvFile)

# Loading and Compiling Model
MODEL = load_model('inception_v3_aw1_0.8104.h5')
MODEL.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Path of image you want to predict
for bodyPart in os.listdir('./images/test'):
    for imageFile in os.listdir('./images/test/'+bodyPart):
        # Find out real class
        realClass = bodyPart
        # Convert Img to an appropriate numpy array
        IMG = image.load_img('./images/test/'+bodyPart+'/'+imageFile, target_size=(299, 299))
        X = image.img_to_array(IMG)
        X = np.expand_dims(X, axis=0)
        IMAGES = np.vstack([X])

        # The actual prediction
        CLASSES = MODEL.predict(IMAGES, batch_size=10)

        # Sorting thr ptrdictions in descending order
        sorting = (-CLASSES).argsort()

        # Getting the top 3 predicitons
        sorted_ = sorting[0][:3]

        # Converting result of prediction to readable categories
        CATEGORIES = {0: 'anal', 1: 'arms', 2: 'armsAndHands',
                3: 'face', 4: 'feet', 5: 'genitalsFemale',
                6: 'genitalsMale', 7: 'hands', 8: 'head',
                9: 'legs', 10: 'legsAndfeet', 11: 'torsoback',
                12: 'torsoFront'}
        RESPONSE = [CATEGORIES[i] for i in sorted_]
        #RESPONSE = ""
        print('Image {}: predict1 {}, predict2 {}, predict3 {}, real {}'.format(imageFile, RESPONSE[0], RESPONSE[1], RESPONSE[2], realClass))
        csvWriter.writerow([imageFile, RESPONSE[0], RESPONSE[1], RESPONSE[2], realClass])
