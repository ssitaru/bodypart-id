#!/usr/bin/python

# this scripts predicts the body part for all images in a given SQL table, and writes them back to SQL

import json
import sys
import mysql.connector
from datetime import datetime
from dateutil.parser import *

import os
import os.path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.preprocessing import image

# Loading and Compiling Model
MODEL = load_model('inception_v3_0.9510416388511658.h5')
MODEL.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

con = mysql.connector.connect(host = "192.168.0.11", user="", password="", database="")
cursor = con.cursor()

cursor.execute("SELECT p.id,p.path FROM photo p LEFT JOIN photoextended pe ON (p.id = pe.photoid) WHERE pe.bodypart LIKE 'other'")

for photo in cursor.fetchall():
    IMG_PATH = '/data/patientenfotos/all/'+os.path.basename(photo[1])
    if(not os.path.isfile(IMG_PATH)):
        #print(IMG_PATH, "does not exist")
        continue
    # Convert Img to an appropriate numpy array
    try:
        IMG = image.load_img(IMG_PATH, target_size=(299, 299))

        X = image.img_to_array(IMG)
        X = np.expand_dims(X, axis=0)
        IMAGES = np.vstack([X])

        # The actual prediction
        CLASSES = MODEL.predict_classes(IMAGES, batch_size=10)

        # Converting result of prediction to readable categories
        CATEGORIES = {0: 'anal', 1: 'arms', 2: 'armsAndHands',
              3: 'face', 4: 'feet', 5: 'genitalsFemale',
              6: 'genitalsMale', 7: 'hands', 8: 'head',
              9: 'legs', 10: 'legsAndfeet', 11: 'torso'}
        RESPONSE = [CATEGORIES[i] for i in CLASSES]
        print(IMG_PATH, "is", RESPONSE[0])

        insCursor = con.cursor(prepared=True)
        sql = "UPDATE photoextended SET bodypart = ? WHERE photoid = ?"
        data = (RESPONSE[0], int(photo[0]))
        insCursor.execute(sql, data)
        con.commit()
        insCursor.close()
    except Exception as e:
        print(IMG_PATH, "error", e)


