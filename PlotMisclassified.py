import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras import backend as K
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import keras_tuner as kt
from tensorflow.keras import regularizers
import random
import gc
import glob
import pickle
import time
import CAIRD
import CAIRD_ML
import CAIRDIngest

# Function to plot images with their predicted and true labels
# Function to plot the Red, Green, and Blue channels of misclassified images
def plot_misclassified_images(images, true_labels, predicted_labels, class_names, n=8):
    misclassified = (true_labels != predicted_labels)
    misclassified_images = images[misclassified][:n]
    misclassified_true_labels = true_labels[misclassified][:n]
    misclassified_predicted_labels = predicted_labels[misclassified][:n]

    plt.figure(figsize=(15, 5 * n))
    for i in range(n):
        # Get the i-th misclassified image and its channels
        img = misclassified_images[i]
        img_r, img_g, img_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Plot the Red channel
        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(img_r, cmap=plt.cm.binary_r)
        plt.title(f"Science\nTrue: {class_names[misclassified_true_labels[i]]}\n"
                  f"Pred: {class_names[misclassified_predicted_labels[i]]}")
        plt.axis("off")

        # Plot the Green channel
        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(img_g, cmap=plt.cm.binary_r)
        plt.title("Reference")
        plt.axis("off")

        # Plot the Blue channel
        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(img_b, cmap=plt.cm.binary_r)
        plt.title("Difference")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

FullData = tf.data.Dataset.load(os.path.join(CAIRD.DatabaseDir, "TF/CurrentDataset"))
print("Loaded DS")
DatasetSize = tf.data.experimental.cardinality(FullData).numpy()

TrainSize = int(0.8 * DatasetSize)
ValSize = int(0.1 * DatasetSize)
TestSize = DatasetSize - TrainSize - ValSize

TrainData = FullData.take(TrainSize)
ValData = FullData.skip(TrainSize).take(ValSize)
TestData = FullData.skip(TrainSize + ValSize)

dataset = TestData

model = CAIRD_ML.ConvNeXt()
model.load_weights("NewTraining.weights.h5")
CAIRD_ML.CompileNN(model)

# Define class names for easier interpretation
class_names = ["Bogus", "SN", "VarStar", "Limit"]
"""
# Evaluate the model on the dataset
all_images = []
all_true_labels = []
all_predicted_labels = []

for images, labels in dataset:
    all_images.append(images)
    all_true_labels.append(labels)
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    all_predicted_labels.append(predicted_labels)

# Convert lists to numpy arrays
all_images = np.concatenate(all_images)
all_true_labels = np.concatenate(all_true_labels)
all_predicted_labels = np.concatenate(all_predicted_labels)

np.save("all_images.npy", all_images)
np.save("all_true_labels.npy", all_true_labels)
np.save("all_predicted_labels.npy", all_predicted_labels)
"""
all_images = np.load("all_images.npy")
all_true_labels = np.load("all_true_labels.npy")
all_predicted_labels = np.load("all_predicted_labels.npy")

# Define the class to inspect for misclassification
true_class = 1 # Actual class
pred_class = 0 # Predicted class

# Filter misclassifications
misclassified_indices = (all_true_labels == true_class) & (all_predicted_labels == pred_class)
misclassified_images = all_images[misclassified_indices]

# Plot some of the misclassified images
plot_misclassified_images(misclassified_images, 
                          all_true_labels[misclassified_indices], 
                          all_predicted_labels[misclassified_indices], 
                          class_names)


TestLabels = np.concatenate([y for x, y in TestData], axis=0)

conf_matrix = tf.math.confusion_matrix(
    TestLabels, all_predicted_labels, num_classes=4)
conf_matrix = tf.cast(conf_matrix, dtype=tf.float32)

# Normalize the confusion matrix
row_sums = tf.reduce_sum(conf_matrix, axis=1)[
    :, tf.newaxis]  # Calculate row sums
normalized_conf_matrix = conf_matrix / row_sums
percentage_conf_matrix = normalized_conf_matrix * 100  # Convert to percentages
percentage_conf_matrix = percentage_conf_matrix.numpy()
plt.figure(figsize=(10, 7))
sns.heatmap(percentage_conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
print(percentage_conf_matrix)

plt.title('Confusion Matrix')
plt.close()
gc.collect()
