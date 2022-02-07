"""
    Python Script for using Model to predict categories of single or several
     images at once
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.preprocessing import image
import csv
import re

import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Get prediction on validation dataset')
parser.add_argument("net", help="net to use")
ARGS = parser.parse_args()

#csv
csvFile = open('predict_top1.csv', 'a', newline="")
csvWriter = csv.writer(csvFile)

# Loading and Compiling Model
MODEL = load_model(ARGS.net)
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
        CLASSES = MODEL(IMAGES, training=False)

        # Converting result of prediction to readable categories
        CATEGORIES = {0: 'anal', 1: 'arms', 2: 'armsAndHands',
              3: 'face', 4: 'feet', 5: 'genitalsFemale',
              6: 'genitalsMale', 7: 'hands', 8: 'head',
              9: 'legs', 10: 'legsAndfeet', 11: 'torso'}
        i=0
        other=True
        max=0
        maxClass=''
        for c in tf.unstack(CLASSES, axis=1):
            #print(CATEGORIES[i], ':  {f:.3f}'.format(f=float(c[0])))
            if(float(c[0]) > max):
                max = float(c[0])
                maxClass = CATEGORIES[i]
            if(float(c[0]) > 0.5):
                other=False
            i += 1
        #print('other:', other)
        if other:
            maxClass = 'other'
        match = 0
        if maxClass == realClass:
            match = 1
        print('Image {}: predict {}, real {}'.format(imageFile, maxClass, realClass))
        csvWriter.writerow([imageFile, maxClass, realClass, match])
