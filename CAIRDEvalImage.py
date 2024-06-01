# Author: Aidan Martas

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

import CAIRDCore as CAIRD

DatabaseDir = "/home/entropian/Documents/DLT40/CAIRD/CAIRDDatasets/"
MLDir = "/home/entropian/Documents/DLT40/CAIRD/CAIRDML/"

def EvaluateImage()

StackedImg, MetadataInst, badcount, IsBad = CAIRD.ImageStacker()

CAIRD.BuildCAIRD(MLDir, DatabaseDir)

