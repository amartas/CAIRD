#!/usr/bin/env python
# Author: Aidan Martas

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

import CAIRDIngest
import CAIRD

gc.enable()
ClassNames = ["Artifact", "SN"]
randnum = CAIRD.Randnum
strategy = tf.distribute.MirroredStrategy()


class PerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, Dataset, model_name):
        self.Dataset = Dataset
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, model_name, logs={}):
        
        TestLabels = np.concatenate([y for x, y in self.Dataset], axis=0)
        
        #self.model.save("CAIRDTraining_" + str(epoch) + "_" + str(model_name) + ".keras")
        self.model.save_weights(os.path.join(CAIRD.MLDir, "CAIRDTraining_" + str(epoch) + "_" + str(model_name) + ".weights.h5"))
        Predictions = self.model.predict(self.Dataset)
        TestLoss, TestAcc = self.model.evaluate(self.Dataset, verbose=2)
        print("Evaluated test images for callback")
        print("TestLoss:", TestLoss)
        print("TestAcc:", TestAcc)
        #for i in range(len(Predictions)):
        #    if Predictions[i][1] > 0.2:
        #        Predictions[i][1] = 1
        
        PredictedClasses = np.argmax(Predictions, axis=1)
    
        conf_matrix = tf.math.confusion_matrix(
            TestLabels, PredictedClasses, num_classes=2)
        conf_matrix = tf.cast(conf_matrix, dtype=tf.float32)
    
        # Normalize the confusion matrix
        row_sums = tf.reduce_sum(conf_matrix, axis=1)[
            :, tf.newaxis]  # Calculate row sums
        normalized_conf_matrix = conf_matrix / row_sums
        percentage_conf_matrix = normalized_conf_matrix * 100  # Convert to percentages
        percentage_conf_matrix = percentage_conf_matrix.numpy()
        plt.figure(figsize=(10, 7))
        sns.heatmap(percentage_conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=ClassNames, yticklabels=ClassNames)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
        print(percentage_conf_matrix)

        plt.title(f'Confusion Matrix - Epoch: {epoch}')
        plt.savefig('TrainingGraphs/'+self.model_name+"_"+str(epoch))
        plt.close()
        gc.collect()



def ConvNeXt():
    model = tf.keras.applications.ConvNeXtTiny(
                    model_name="convnext_tiny",
                    include_top=True,
                    include_preprocessing=True,
                    weights=None,
                    input_tensor=None,
                    input_shape=(50,50,3),
                    pooling="max",
                    classes=2,
                    classifier_activation="softmax",
                    )
    return model

def CompileNN(model):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = ["accuracy"]
        )

def DataGenerator(images, labels, batch_size):
    num_samples = images.shape[0]
    while True:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            yield images[start:end], labels[start:end]

def create_tf_dataset(images, labels, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: DataGenerator(images, labels, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 50, 50, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )
    )
    return dataset

"""

BuildCAIRD(MLDir, DatabaseDIR)

The core ML training and definition.  Treat with caution.

MLDir       - Directory within CAIRD's file structure for ML files

DatabaseDir - Directory within CAIRD's file structure for image datasets

"""

def BuildCAIRD(MLDir, DatabaseDir):

    plt.rcParams['figure.dpi'] = 300
    
    
    def PlotPhotos(i, img, channel, prediction, classification, label, ClassNames):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img[channel], cmap=plt.cm.binary)
            
        plt.xlabel("{} ({})".format(ClassNames[prediction],
                                             classification))
    
    BatchSize = 64
    
    FullData = tf.data.Dataset.load(os.path.join(CAIRD.DatabaseDir, "TF/CurrentDataset"))
    
    DatasetSize = tf.data.experimental.cardinality(FullData).numpy()
    
    TrainSize = int(0.8 * DatasetSize)
    ValSize = int(0.1 * DatasetSize)
    TestSize = DatasetSize - TrainSize - ValSize
    
    TrainData = FullData.take(TrainSize)
    ValData = FullData.skip(TrainSize).take(ValSize)
    TestData = FullData.skip(TrainSize + ValSize)

    
    #TrainData = TrainData.batch(BatchSize).prefetch(tf.data.experimental.AUTOTUNE)
    #ValData = ValData.batch(BatchSize).prefetch(tf.data.experimental.AUTOTUNE)
    #TestData = TestData.batch(BatchSize).prefetch(tf.data.experimental.AUTOTUNE)
    
    TrainLabels = np.concatenate([y for x, y in TrainData], axis=0)
    print(len(TrainLabels), "TL LENGTH")
    objecttype, counts = np.unique(TrainLabels, return_counts = True)
    
    ClassWeights = {0: np.max(counts)/counts[0], # Normalize training dataset weights
                    1: np.max(counts)/counts[1]
                    }
    del TrainLabels, objecttype, counts
    print(ClassWeights, "Class Weights")
    
    gc.collect()
    
    model = ConvNeXt()
    CompileNN(model)
    
    PerfCallback = PerformancePlotCallback(ValData, "ConvNeXtTiny")
    
    history = model.fit(TrainData,
                        validation_data = ValData,
                        epochs=50,
                        callbacks=[PerfCallback],
                        class_weight = ClassWeights,
                        batch_size = BatchSize
                        )
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    
    model.save_weights(os.path.join(CAIRD.MLDir, "FullEpoch"))
    
    del model
    
    model = ConvNeXt()
    CompileNN(model)
    
    history = model.fit(TrainData,
                        validation_data = ValData,
                        epochs=best_epoch,
                        callbacks=[PerfCallback],
                        class_weight = ClassWeights,
                        batch_size = BatchSize
                        )
    model.save_weights(os.path.join(CAIRD.MLDir, "CompleteTrainingWeights"))
    Predictions = model.predict(TestData)
    print(Predictions)
    TestLoss, TestAcc = model.evaluate(TestData, verbose=2)
    
    # Get the class with the highest probability
    
    print("\nTest accuracy:", TestAcc)
    
    plt.hist(Predictions[0], bins=20)
    plt.xlabel("Confidence in class")
    plt.ylabel("Number of images")
    plt.show()




"""

ClassifyImage(img, metadata)

Takes the image and and metadata as inputs and returns classification confidences.

"""

def ClassifyImage(scipath, refpath, diffpath, outputdir, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax): # I cannot describe how good it feels to finally have this function
    
    InputImg, InputMD = CAIRDIngest.InputProcessor(scipath, refpath, diffpath, "", xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)
    
    InputMD = np.asarray(InputMD)

    InputImg, InputMD = np.array([InputImg]), np.array([InputMD])

    model = ConvNeXt()
    model.load_weights("NewTraining.weights.h5")
    CompileNN(model)
    Prediction = model.predict(InputImg)
    PredictionDict = {"Bogus": Prediction[0][0],
                      "SN": Prediction[0][1]}
    return PredictionDict

#BuildCAIRD(CAIRD.MLDir, CAIRD.DatabaseDir)















































