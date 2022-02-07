#!/usr/bin/python

# helper file to get files based on DB entries

import json
import sys
import os
import mysql.connector
from datetime import datetime
from dateutil.parser import *
from shutil import copy

# SELECT DISTINCT(bodypart) FROM photoextended WHERE manuallyTagged = 1 
con = mysql.connector.connect(host = "192.168.0.11", user="root", password="", database="")
cursor = con.cursor()

cursor.execute("SELECT DISTINCT(bodypart) FROM photoextended WHERE manuallyTagged = 1 ")

for bodypart in cursor.fetchall():
    bp = bodypart[0]
    try:
        os.mkdir('images.unsorted/'+bp)
    except FileExistsError:
        pass
    crs = con.cursor()
    crs.execute("SELECT p.path FROM photoextended pe JOIN photo p ON (pe.photoid = p.id) WHERE manuallyTagged = 1 AND bodypart = '" + bp + "'")
    for photo in crs.fetchall():
        IMG_PATH = '/data/patientenfotos/all/'+os.path.basename(photo[0])
        if(not os.path.isfile(IMG_PATH)):
            print(IMG_PATH, "does not exist")
            continue
        copy(IMG_PATH, 'images.unsorted/'+bp+'/')
        print('copied '+IMG_PATH+' to '+'images.unsorted/'+bp+'/')
