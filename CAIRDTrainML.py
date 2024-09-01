#!/usr/bin/env python
# Author: Aidan Martas

import os
import CAIRD
import CAIRDIngest

import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import Sequence
from keras import backend as K
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import keras_tuner as kt
from tensorflow.keras import regularizers
import random
import gc
from glob import glob
import pickle

DatabaseDir = CAIRD.DatabaseDir
MLDir = CAIRD.MLDir
strategy = tf.distribute.MirroredStrategy()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Ingest training data, preprocess it, and then build a new CAIRD model
#CAIRDIngest.DatasetGenerator()
CAIRD.BuildCAIRD(MLDir, DatabaseDir)