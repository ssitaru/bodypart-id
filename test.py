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

#csv
csvFile = open('delta.csv', 'a', newline="")
csvWriter = csv.writer(csvFile)


# Loading and Compiling Model
MODEL = load_model('inception_v3_0.9635416865348816.h5')
MODEL.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Path of image you want to predict
for imageFile in os.listdir('./tests/images/'):
    # Find out real class
    realClass = re.sub("([a-zA-Z]+)\-(\d+).jpg", r"\1", imageFile)


    # Convert Img to an appropriate numpy array
    IMG = image.load_img('./tests/images/'+imageFile, target_size=(299, 299))
    X = image.img_to_array(IMG)
    X = np.expand_dims(X, axis=0)
    IMAGES = np.vstack([X])

    # The actual prediction
    CLASSES = MODEL.predict(IMAGES, batch_size=10)


    #if(CLASSES[0][CLASSES.argmax(axis=1)] < 0.1):
    #   print('Predicted Classes for Images: others')
    #else:
        # Converting result of prediction to readable categories
    CATEGORIES = {0: 'anal', 1: 'arms', 2: 'armsAndHands',
              3: 'face', 4: 'feet', 5: 'genitalsFemale',
              6: 'genitalsMale', 7: 'hands', 8: 'head',
              9: 'legs', 10: 'legsAndfeet', 11: 'torso'}
    #RESPONSE = [CATEGORIES[i] for i in CLASSES[0]]

    # delta: max value - mean value
    maxV = CLASSES[0][CLASSES.argmax()]
    newClassesWithoutMax = np.delete(CLASSES[0], CLASSES.argmax())
    print('Predicted Classes for Images: {}'.format(CATEGORIES[CLASSES.argmax()]))
    print("max prediction is", maxV)
    print("delta is", maxV - newClassesWithoutMax.mean())
    csvWriter.writerow([imageFile, realClass, CATEGORIES[CLASSES.argmax()], maxV, maxV - newClassesWithoutMax.mean()])