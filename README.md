# bodypart-id

An algorithm for classifying dermatological clinical images to bodyparts. This algorithm uses keras and tensorflow. The code is licensed under a MIT license.

Code authors: Sebastian Sitaru, Talel Oueslati

## Install

Install python v3 and pip. Then run:

```
pip install -r requirements.txt
mkdir images.unsorted
mkdir -p images/{train,test}
```

## Preparing data

Put the data in a folder e.g., `images.unsorted` (the folder should not be named `images`). The data should be sorted as follows:
- images.unsorted/`bodypart1`/
- images.unsorted/`bodypart1`/image1.jpg
- images.unsorted/`bodypart1`/image2.jpg (exact filename of image is not relevant)
- images.unsorted/`bodypart1`/`...`
- images.unsorted/`bodypart2`/
- images.unsorted/`bodypart2`/image1.jpg
- images.unsorted/`bodypart2`/image2.jpg
- `...`

The original dataset cannot be provided, since it contains sensible patient photos. To test, a dataset like [https://www.tensorflow.org/datasets/catalog/cats_vs_dogs](Cats & Dogs) can be used.

To split the data into test/train datasets, run:
```
python shuffle_split.py images.unsorted
```

This will sort the data into `images/{train,test}/...`.

## Training
```
usage: x.py [-h] -p DATA_PATH [-w] [-a] [-s STEPS_PER_EPOCH] [-e EPOCHS] [-l LEARNING_RATE] runid

Train x

positional arguments:
  runid                 run ID

optional arguments:
  -h, --help            show this help message and exit
  -p DATA_PATH, --data-path DATA_PATH
                        Path to data e.g. images/{test,train}
  -w, --no-init-weights
                        No init weights from ImageNet
  -a, --data-augmentation
                        Use data augmentation
  -s STEPS_PER_EPOCH, --steps-per-epoch STEPS_PER_EPOCH
                        Steps per epoch
  -e EPOCHS, --epochs EPOCHS
                        Epochs
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
```

For training, you could eg. run:
```
python <net>.py -p images 001
```

This will generate a file in the current directory: `./xception_{runid}_{validation_accuracy}.h5`

## Predicting

For prediction, multiple .py scripts are supplied. The most basic one is `predict.py`.

```
usage: predict.py [-h] net image

Predict on single image

positional arguments:
  net         net to use
  image       image

optional arguments:
  -h, --help  show this help message and exit
```