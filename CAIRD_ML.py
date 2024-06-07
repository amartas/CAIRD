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
import pickle

import CAIRDIngest
import CAIRD

gc.enable()
ClassNames = ["Artifact", "SN", "VS"]
randnum = CAIRD.Randnum


class PerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, TestImgs, TestMetadata, TestLabels, model_name):
        self.TestImgs = TestImgs
        self.TestMetadata = TestMetadata
        self.TestLabels = TestLabels
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, model_name, logs={}):

        self.model.save("CAIRDTraining_" + str(epoch) + "_" + str(model_name) + ".keras")
        
        Predictions = self.model.predict([self.TestImgs, self.TestMetadata])
        TestLoss, TestAcc = self.model.evaluate([self.TestImgs, self.TestMetadata], self.TestLabels, verbose=2)
        
        #for i in range(len(Predictions)):
        #    if Predictions[i][1] > 0.2:
        #        Predictions[i][1] = 1
                
        
        PredictedClasses = np.argmax(Predictions, axis=1)
        
        # Get correctly/incorrectly classified indices
        CorrectIndices = np.nonzero(PredictedClasses == self.TestLabels)[0] 
        IncorrectIndices = np.nonzero(PredictedClasses != self.TestLabels)[0]
    
        conf_matrix = tf.math.confusion_matrix(
            self.TestLabels, PredictedClasses, num_classes=3)
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
    
    
    ArrInfoFile = open(os.path.join(CAIRD.DatabaseDir, "Arrays/DatasetInfo.dict"), "r")
    DatabaseInfo = ArrInfoFile.read()
    ArrInfoFile.close()
    DatabaseInfo = eval(DatabaseInfo)
    
    print(DatabaseInfo)
    
    ImgDatabase = np.memmap(os.path.join(DatabaseDir, "Arrays/ImgDatabase.memmap"), mode = "r", dtype = "float64", shape = DatabaseInfo["ImgDatabase"])
    ImgLabels = np.memmap(os.path.join(DatabaseDir, "Arrays/ImgLabels.memmap"), mode = "r", dtype = "float64", shape = DatabaseInfo["ImgLabels"])
    ImgMetadata = np.memmap(os.path.join(DatabaseDir, "Arrays/ImgMetadata.memmap"), mode = "r", dtype = "float64", shape = DatabaseInfo["ImgMetadata"])
    

    print(ImgDatabase.shape, "Database shape")
    print(ImgLabels.shape)
    # Drove my Chevy to the levee but the levee was dry...
    print(np.nanmax(ImgDatabase))
    print(ImgMetadata.shape)

    objecttype, counts = np.unique(ImgLabels, return_counts = True)
    
    ClassWeights = {0: np.max(counts)/counts[0], # Normalize training dataset weights
                    1: np.max(counts)/counts[1],
                    2: np.max(counts)/counts[2]
                    }
    
    
    print(ClassWeights, "Class Weights")
    
    
    
    TrainImgs, TestImgs, TrainLabels, TestLabels, TrainMetadata, TestMetadata = train_test_split(ImgDatabase, ImgLabels, ImgMetadata, random_state=randnum, test_size=0.05)
    
    print(TrainImgs.shape)
    print(TrainLabels.shape)
    print(TrainMetadata.shape)
    print(TestImgs.shape)
    print(TestLabels.shape)
    print(TestMetadata.shape)
    
    
    
    print(ClassWeights)
    
    del ImgDatabase, ImgLabels, ImgMetadata # Save memory
    gc.collect()
    
    strategy = tf.distribute.MirroredStrategy()
    
    def ModelBuilder(hp):
    
        # Hyperparameter optimization parameters
        HPDenseUnits = hp.Int("DenseUnits", min_value = 32, max_value = 256, step = 16)
        HPConvUnits = hp.Int("ConvUnits", min_value = 32, max_value = 256, step = 16)
        HPRegularization = hp.Float("RegParam", min_value = 0.4, max_value = 0.6, step = 0.1)
        HPDropout = hp.Float("Dropout", min_value = 0.4, max_value = 0.6, step = 0.1)
        HPLearningRate = hp.Choice("LearningRate", values = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5])
    
        # Convolutional layers
        SmallConvInput = tf.keras.Input(shape = (51, 51, 3))
        x = layers.Conv2D(HPConvUnits/2, 3, activation = "relu", data_format="channels_last")(SmallConvInput)
        x = layers.Conv2D(HPConvUnits/2, 3, activation = "relu", data_format="channels_last")(x)
        x = layers.Conv2D(HPConvUnits/2, 3, activation = "relu", data_format="channels_last")(x)
        x = layers.MaxPooling2D(2, data_format="channels_last")(x)
        x = layers.Conv2D(HPConvUnits, 3, activation = "relu", data_format="channels_last")(x)
        x = layers.Conv2D(HPConvUnits, 3, activation = "relu", data_format="channels_last")(x)
        x = layers.Conv2D(HPConvUnits, 3, activation = "relu", data_format="channels_last")(x)
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.Dense(64, activation = "relu", kernel_regularizer=regularizers.l2(0.6))(x)
        SmallConvOutput = layers.Dropout(HPDropout)(x)
        
        CNNSection = tf.keras.Model(SmallConvInput, SmallConvOutput, name = "SmallConv")
        CNNSection.summary()
        
        # Metadata concatenation
        MetadataInput = tf.keras.Input(shape = (6,))
        MetadataNormalized = layers.BatchNormalization()(MetadataInput)
        
        model = layers.Concatenate()([CNNSection.output, MetadataNormalized])
        model = layers.Dense(HPDenseUnits, activation = "relu", kernel_regularizer=regularizers.l2(HPRegularization))(model)
        model = layers.Dropout(HPDropout)(model)
        model = layers.Dense(HPDenseUnits, activation = "relu", kernel_regularizer=regularizers.l2(HPRegularization))(model)
        ModelOutput = layers.Dense(3, activation = "softmax")(model)
        
        model = tf.keras.Model(
            inputs=[CNNSection.input, MetadataInput],
            outputs=ModelOutput
        )
        
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = HPLearningRate),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            metrics = ["accuracy"]
            )
      
        return model
    
    tuner = kt.Hyperband(
        ModelBuilder,
        objective='val_accuracy',
        max_epochs=20,
        factor = 3,
        directory="CAIRD_HP",
        project_name="CAIRDLatestTraining",
        hyperband_iterations = 5
        )
    
    StopEarly = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 5)
    
    tuner.search([TrainImgs, TrainMetadata], TrainLabels, epochs = 50, validation_split = 0.1, callbacks = [StopEarly], class_weight=ClassWeights)
    
    BestParams = tuner.get_best_hyperparameters(num_trials = 1)[0]
    
    print(BestParams.values, "PARAMS")
    
    PerfCallback = PerformancePlotCallback(TestImgs, TestMetadata, TestLabels, "HP")
    
    model = tuner.hypermodel.build(BestParams)
    model.summary()
    
    PerfCallback = PerformancePlotCallback(TestImgs, TestMetadata, TestLabels, "HP")
    
    gc.collect()
    
    model = ModelBuilder(BestParams)
    
    history = model.fit([TrainImgs, TrainMetadata], TrainLabels, epochs=50, validation_split=0.1, callbacks=[PerfCallback])
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    
    del model, history
    gc.collect()
    
    hypermodel = tuner.hypermodel.build(BestParams)
    
    print(BestParams.values, "PARAMS")
    
    del tuner
    gc.collect()
    
    hypermodel.fit([TrainImgs, TrainMetadata], TrainLabels, epochs=best_epoch, validation_split=0.1)
    
    hypermodel.save(os.path.join(MLDir, "CAIRDLatestTraining.keras"))
    
    hypermodel = tf.keras.models.load_model(os.path.join(MLDir, "CAIRDLatestTraining.keras"))
    
    Predictions = hypermodel.predict([TestImgs, TestMetadata])
    TestLoss, TestAcc = hypermodel.evaluate([TestImgs, TestMetadata], TestLabels, verbose=2)
    
    # Get the class with the highest probability
    
    print("\nTest accuracy:", TestAcc)
    
    
    plt.hist(Predictions, bins=10)
    plt.xlabel("Confidence in class")
    plt.ylabel("Number of images")
    plt.show()


"""

ClassifyImage(img, metadata)

Takes the image and and metadata as inputs and returns classification confidences.

"""

def ClassifyImage(scipath, refpath, diffpath, outputdir, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax): # I cannot describe how good it feels to finally have this function
    
    img, metadata = CAIRDIngest.InputProcessor(scipath, refpath, diffpath, outputdir, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)

    img = np.expand_dims(img, axis=0)
    metadata = np.expand_dims(metadata, axis=0)
    
    model = tf.keras.models.load_model(os.path.join(CAIRD.MLDir, "CAIRDLatestTraining.keras"))
    Prediction = model.predict([img, metadata])
    PredictionDict = {"Artifact": Prediction[0][0],
                      "SN": Prediction[0][1],
                      "VarStar": Prediction[0][2]}
    return PredictionDict
