"""
    Python Script for using Model to predict categories of single or several
     images at once
"""
import os

import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.preprocessing import image
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Predict on single image')
parser.add_argument("net", help="net to use")
parser.add_argument("image", help="image")
ARGS = parser.parse_args()

# Loading and Compiling Model
MODEL = load_model(ARGS.net)
MODEL.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Path of image you want to predict
IMG_PATH = ARGS.image

# Convert Img to an appropriate numpy array
IMG = image.load_img(IMG_PATH, target_size=(224, 224))
X = image.img_to_array(IMG)
X = np.expand_dims(X, axis=0)
IMAGES = np.vstack([X])

# The actual prediction
CLASSES = MODEL(IMAGES, training=False)

# Converting result of prediction to readable categories
CATEGORIES = {0: 'anal', 1: 'arms', 2: 'armsAndHands',
              3: 'face', 4: 'feet', 5: 'genitalsFemale',
              6: 'genitalsMale', 7: 'hands', 8: 'head',
              9: 'legs', 10: 'legsAndFeet', 11: 'torso'}

print('Predicted Classes for Images: ')
i=0
other=True
for c in tf.unstack(CLASSES, axis=1):
    print(CATEGORIES[i], ':  {f:.3f}'.format(f=float(c[0])))
    i += 1
    if(float(c[0]) > 0.5):
        other=False

print('other:', other)
