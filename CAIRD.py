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


BaseDir = "/home/entropian/Documents/CAIRD/"

DatabaseDir = os.path.join(BaseDir, "CAIRDDatasets/")
MLDir = os.path.join(BaseDir, "CAIRDML/")
TempStorageDir = os.path.join(BaseDir, "TempStorage/")

Randnum = 242

import CAIRDIngest
import CAIRD_ML

def DatasetPreparation():
    CAIRDIngest.DatasetPreparation(51)

def BuildCAIRD(MLDir, DatabaseDir):
    CAIRD_ML.BuildCAIRD(MLDir, DatabaseDir)

def Reviewer(inputdir, n_imgs, outputdir):
    CAIRDIngest.Reviewer(inputdir, n_imgs, outputdir)
    
def InputProcessor(scipath, refpath, diffpath, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax):
    CAIRDIngest.InputProcessor(scipath, refpath, diffpath, TempStorageDir, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)
    
def ClassifyImage(scipath, refpath, diffpath, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax):
    return CAIRD_ML.ClassifyImage(scipath, refpath, diffpath, TempStorageDir, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)