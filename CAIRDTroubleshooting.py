import CAIRD
import CAIRDIngest
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras import backend as K
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import keras_tuner as kt
from tensorflow.keras import regularizers
import random
import gc
from glob import glob

scipath = "/home/entropian/Documents/CAIRD/CAIRDDatasets/SortedData/SN/1181.clean.fits"
refpath = "/home/entropian/Documents/CAIRD/CAIRDDatasets/SortedData/SN/1181.ref.fits"
diffpath = "/home/entropian/Documents/CAIRD/CAIRDDatasets/SortedData/SN/1181.diff.fits"

CID = None
TID = None
RA = None
DEC = None
fluxrad = None
ellipticity = None
fwhm = None
bkg = None
fluxmax = None

"""
InputImg, InputMD = CAIRDIngest.InputProcessor(scipath, refpath, diffpath, "", 25, 25, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)

InputMD = np.asarray(InputMD)

InputImg, InputMD = np.array([InputImg]), np.array([InputMD])

model = tf.keras.models.load_model(os.path.join(CAIRD.MLDir, "CAIRDLatestTraining.keras"))
Prediction = model.predict([InputImg, InputMD])
"""

Prediction = CAIRD.ClassifyImage(scipath, refpath, diffpath, 25, 25, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)

print(Prediction)