## bodypart-id

An algorithm for classifying dermatological clinical images to bodyparts. This algorithm uses keras and tensorflow. The code is licensed under a MIT license.

# Install

Install python v3 and pip. Then run:

```
pip install -r requirements.txt
mkdir images.unsorted
```

# Training

Put the data in a folder e.g., `images.unsorted` (the folder should not be named `images`). The data should be sorted as follows:
- images.unsorted/`bodypart1`/
- images.unsorted/`bodypart1`/image1.jpg
- images.unsorted/`bodypart1`/image2.jpg (exact filename of image is not relevant)
- images.unsorted/`bodypart1`/`...`
- images.unsorted/`bodypart2`/
- images.unsorted/`bodypart2`/image1.jpg
- images.unsorted/`bodypart2`/image2.jpg
- `...`

The original dataset cannot be provided, since it contains sensible patient photos. To test, a dataset like (Cats&Dogs)[https://www.tensorflow.org/datasets/catalog/cats_vs_dogs] can be used.

To split the data into test/train datasets, run:
```
python shuffle_split.py images.unsorted
```

This will sort the data into `images/{train,test}/...`.