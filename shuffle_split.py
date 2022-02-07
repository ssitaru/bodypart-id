"""
    Python script for splitting image data to train and test data per category/bodypart
"""
import os
import argparse
import random
from shutil import copy

# Parse first argument which is bodypart (e.g. python shuffle_split.py arms)
PARSER = argparse.ArgumentParser(description='Split images.')
PARSER.add_argument('path', metavar='P', type=str, nargs='+',
                    help='path')
ARGS = PARSER.parse_args()
PATH = ARGS.path[0]
print(PATH)

# Create IMGS array with all images of one category (you might have to batch if you have too many)
IMGS = []
VALID_IMAGES = [".jpg", ".jpeg"]
for bp in os.listdir(PATH):
    #ext = os.path.splitext(f)[1]
    #if ext.lower() not in VALID_IMAGES:
    #    continue
    IMGS = []
    print('bodypart', bp)
    
    for f in os.listdir(os.path.join(PATH, bp)):
        IMGS.append(os.path.join(PATH, bp, f))
        #print(os.path.join(PATH, bp, f))

    # Split images in train and test data
    random.shuffle(IMGS)
    split_1 = int(0.8*len(IMGS))
    train_imgs = IMGS[:split_1]
    test_imgs = IMGS[split_1:]
    print(len(IMGS))
    print(len(train_imgs))
    print(len(test_imgs))
    try:
        os.mkdir('images/train/'+bp)
        os.mkdir('images/test/'+bp)
    except FileExistsError:
        pass
    for img in train_imgs:
        copy(img, 'images/train/'+bp+'/'+os.path.basename(img))
        #pass
    for img in test_imgs:
        copy(img, 'images/test/'+bp+'/'+os.path.basename(img))
        #pass 



