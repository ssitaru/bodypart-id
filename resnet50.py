import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import argparse

parser = argparse.ArgumentParser(description='Train resnet')
parser.add_argument("runid", help="run ID", type=int)
parser.add_argument("-p", "--data-path", help="Path to data e.g. images/{test,train}", required=True)
parser.add_argument("-w", "--no-init-weights", action='store_true', default=False, help="No init weights from ImageNet")
parser.add_argument("-a", "--data-augmentation", action='store_true', default=False, help="Use data augmentation")
parser.add_argument("-s", "--steps-per-epoch", default=80, help="Steps per epoch")
parser.add_argument("-e", "--epochs", default=80, help="Epochs")
parser.add_argument("-l", "--learning-rate", default=2e-5, help="Learning rate")
ARGS = parser.parse_args()
dp = ARGS.data_path


# Convert Images to tensors suitable for the model
if ARGS.data_augmentation:
      TRDATA = ImageDataGenerator(
            zoom_range=0.15,
            rotation_range=15,
            horizontal_flip=True)
else:
      TRDATA = ImageDataGenerator()
TRAINDATA = TRDATA.flow_from_directory(directory='./'+dp+'/train/',target_size=(224,224))
TSDATA = ImageDataGenerator()
TESTDATA = TSDATA.flow_from_directory(directory='./'+dp+'/test/', target_size=(224,224))

#Setting up model
MODEL = keras.models.Sequential()
# We use the pretrained model ResNet50 provided by keras
if ARGS.no_init_weights:
      w = 'imagenet'
else:
      w = None
MODEL.add(ResNet50(include_top=False, weights=w, input_shape=(224,224,3)))
# Dropout layer to prevent overfitting
MODEL.add(keras.layers.Dropout(rate=0.5))
# Flatten layer for flattening the three dimensions to one single one
MODEL.add(keras.layers.Flatten())
# Dense layer (normal Neurons layer) with (*, 256) arrays as output and activation function relu
MODEL.add(keras.layers.Dense(256, activation='relu'))
# Last Layer is again Dense Layer with (*, 14) and activation function sigmoid
MODEL.add(keras.layers.Dense(12, activation='sigmoid'))
MODEL.summary()

# Compiling model for usage
MODEL.compile(optimizer=keras.optimizers.RMSprop(lr=ARGS.learning_rate),
      loss='categorical_crossentropy',
      metrics=['acc'])

HISTORY = MODEL.fit_generator(TRAINDATA,
                         steps_per_epoch = ARGS.steps_per_epoch,
                         epochs = ARGS.epochs,
                         validation_data = TESTDATA,
                         validation_steps = 30)

valacc = HISTORY.history['val_acc']
MODEL.save('./resnet50_{runid}_{valacc:.4f}.h5'.format(runid=ARGS.runid, valacc=valacc[len(valacc)-1]))
