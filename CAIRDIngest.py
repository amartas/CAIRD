#!/usr/bin/env python
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
import glob
import pickle
import time
import CAIRDExceptions
import CAIRD

DatabaseDir = CAIRD.DatabaseDir


"""

Reviewer(inputdir, n_imgs, outputdir)

Rudimentarily facilitates fast human review of images with matplotlib.

inputdir - Directory of images to be reviewed.  Expects fits files of the format:
    *.clean.fits - Science image
    *.ref.fits   - Reference image
    *.diff.fits  - Difference image

n_imgs - Number of images to be reviewed

outputdir - Directory for classified images to be saved.  Expects /Artifacts, /SN, /VarStars, /Satellites as subdirectories.

Images have a tag applied to their fits header: "REVIEWER", which is set to "HUMAN" upon classification.

"""

def Reviewer(inputdir, n_imgs, outputdir):
    
    def DisplayImg(scipath, refpath, diffpath):
        with fits.open(scipath) as hdu1:
            with fits.open(refpath) as hdu2:
                with fits.open(diffpath) as hdu3:
                    SciData = hdu1[0].data
                    RefData = hdu2[0].data
                    DiffData = hdu3[0].data
                    index = 1
                    for img in [SciData, RefData, DiffData]:
                        plt.subplot(1, 3, index)
                        plt.grid(False)
                        if index == 1:
                            plt.xlabel("Science")
                        if index == 2:
                            plt.xlabel("Reference")
                        if index == 3:
                            plt.xlabel("Difference")
                        plt.xticks([])
                        plt.yticks([])
                        plt.imshow(img, cmap=plt.cm.binary)
                        index += 1
        plt.show(block=False)

    def ImageTagger(scipath, tag, outputpath):
        with fits.open(scipath) as hdu1:
            SciHeader = hdu1[0].header
            ReturnHeader = hdu1[0].header
            SciHeader["REVIEWER"] = tag
            fits.writeto(outputpath, hdu1[0].data, SciHeader, overwrite=True)
            fits.writeto(scipath, hdu1[0].data, SciHeader, overwrite=True)

    def InputChecker():
        while True:
            ident = input("Class [a, v, s, t]: ")
            if ident in ["a", "v", "s", "t"]:
                return ident

    class AlreadyClassified(Exception):
        pass
    
    FPArtifacts = len(glob(outputdir + "Artifacts/*.clean.fits")) + 1
    FPVarStars = len(glob(outputdir + "VarStars/*.clean.fits"))
    FPSN = len(glob(outputdir + "SN/*.clean.fits"))
    FPSat = len(glob(outputdir + "Satellites/*.clean.fits"))
    imgcount = 0
    print(FPArtifacts, FPVarStars, FPSN, FPSat)
    for scipath in glob(inputdir + "*.clean.fits"):
        try:
            with fits.open(scipath) as hdu4:
                prevclass = hdu4[0].header["CLASSID"]
                if prevclass == 8:
                    prevclass = "Artifact"
                elif prevclass == 11:
                    prevclass = "SN"
                elif prevclass == 14:
                    prevclass = "Artifact"
                if "REVIEWER" in hdu4[0].header:
                    if hdu4[0].header["REVIEWER"] == "HUMAN":
                        raise AlreadyClassified
                
            print("Not classified")
            refpath = scipath[:-10] + "ref.fits"
            diffpath = scipath[:-10] + "diff.fits"
            DisplayImg(scipath, refpath, diffpath)
            print("Previously classified as", prevclass)
            if imgcount >= n_imgs:
                print("Classification batch complete")
                return
            print(FPArtifacts, FPVarStars, FPSN, FPSat)
            ident = InputChecker()
            
            if ident == "a":
                FPArtifacts += 1
                ImageTagger(scipath, "HUMAN", outputdir + "Artifacts/" + str(FPArtifacts) + ".clean.fits")
                with fits.open(refpath) as hdu1:
                    fits.writeto(outputdir + "Artifacts/" + str(FPArtifacts) + ".ref.fits", hdu1[0].data, hdu1[0].header, overwrite=False)
                with fits.open(diffpath) as hdu1:
                    fits.writeto(outputdir + "Artifacts/" + str(FPArtifacts) + ".diff.fits", hdu1[0].data, hdu1[0].header, overwrite=False)
                print("Classified as artifact")
            elif ident =="v":
                FPVarStars += 1
                ImageTagger(scipath, "HUMAN", outputdir + "VarStars/" + str(FPVarStars) + ".clean.fits")
                with fits.open(refpath) as hdu1:
                    fits.writeto(outputdir + "VarStars/" + str(FPVarStars) + ".ref.fits", hdu1[0].data, hdu1[0].header, overwrite=False)
                with fits.open(diffpath) as hdu1:
                    fits.writeto(outputdir + "VarStars/" + str(FPVarStars) + ".diff.fits", hdu1[0].data, hdu1[0].header, overwrite=False)
                print("Classified as varstar")
            elif ident =="s":
                FPSN += 1
                ImageTagger(scipath, "HUMAN", outputdir + "SN/" + str(FPSN) + ".clean.fits")
                with fits.open(refpath) as hdu1:
                    fits.writeto(outputdir + "SN/" + str(FPSN) + ".ref.fits", hdu1[0].data, hdu1[0].header, overwrite=False)
                with fits.open(diffpath) as hdu1:
                    fits.writeto(outputdir + "SN/" + str(FPSN) + ".diff.fits", hdu1[0].data, hdu1[0].header, overwrite=False)
                print("Classified as SN")
            elif ident =="t":
                FPSat += 1
                ImageTagger(scipath, "HUMAN", outputdir + "Satellites/" + str(FPSat) + ".clean.fits")
                with fits.open(refpath) as hdu1:
                    fits.writeto(outputdir + "Satellites/" + str(FPSat) + ".ref.fits", hdu1[0].data, hdu1[0].header, overwrite=False)
                with fits.open(diffpath) as hdu1:
                    fits.writeto(outputdir + "Satellites/" + str(FPSat) + ".diff.fits", hdu1[0].data, hdu1[0].header, overwrite=False)
                print("Classified as satellite")
            else:
                print("Parsing failure - skipping")
            imgcount += 1
            pass
        except AlreadyClassified:
            print("Image has already been classified - skipping")
            continue

"""

UniformityCheck(image, edge_width)

Determines whether images were taken at the edge of the chip by detecting if the corners have atypically uniform values (e.g. zeros from filling or padding of the image).

image      - Input image to be checked

edge_width - Length along the edge to check for uniformity

"""

def UniformityCheck(image, edge_width):
    # Check top, bottom, left, right edges
    top_left_corner = image[:edge_width, :edge_width]
    top_right_corner = image[:edge_width, -edge_width:]
    bottom_left_corner = image[-edge_width:, :edge_width]
    bottom_right_corner = image[-edge_width:, -edge_width:]

    # Check if these edges are uniform
    edges = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]
    for edge in edges:
        if np.unique(edge).size == 1:
            raise CAIRDExceptions.CorruptImage("Uniformity check failed")

"""

Stamper(img, output, x_pos, y_pos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)

Stamps an input image to 21x21, pads it to 51x51, and saves it.

img    - Input image (including name and extension)

output - Output filepath (including name and extension)

x_pos  - X-position of the center of the stamp in the source image

y_pos  - Y-position of the center of the stamp in the source image

others - Metadata required for ML to function:
    CID: Classification ID
    TID: Target ID
    RA: Right Ascension
    DEC: Declination
    fluxrad: fluxrad
    ellipticity: ellipticity
    fwhm: fwhm
    bkg: background
    fluxmax: fluxmax
    
    All of these values can be found in the DLT40 MySQL candidates table.

"""

def Stamper(img, output, save, x_pos = 25, y_pos = 25, CID = None, TID = None, RA = None, DEC = None, fluxrad = None, ellipticity = None, fwhm = None, bkg = None, fluxmax = None, DoStamp = True):
    
    if os.path.exists(img) == False:
        print("Missing file")
        raise CAIRDExceptions.CorruptImage("Image does not exist")
    
    with fits.open(img) as hdu1:
        # Extracts the data
        data = hdu1[0].data
        y, x = data.shape
        if DoStamp == True:
            stamp = data[int(y_pos) - 25 : int(y_pos) + 25, int(x_pos) - 25 : int(x_pos) + 25] # Crops the image
        else:
            stamp = data # Doesn't crop the image
        
        # Uniformity check
        UniformityCheck(stamp, 4)
        
        # Updates the FITS header with the dimensions of the stamp
        hdu1[0].header["NAXIS1"] = stamp.shape[1]
        hdu1[0].header["NAXIS2"] = stamp.shape[0]
        
        if CID == None or TID == None or RA == None or DEC == None or fluxrad == None or ellipticity == None or fwhm == None or bkg == None or fluxmax == None:
            CID = hdu1[0].header["CLASSID"]
            TID = hdu1[0].header["DETECTIONID"]
            RA = hdu1[0].header["RA"]
            DEC = hdu1[0].header["DEC"]
            fluxrad = hdu1[0].header["FLUXRAD"]
            ellipticity = hdu1[0].header["ELLIPTICITY"]
            fwhm = hdu1[0].header["FWHM"]
            fluxmax = hdu1[0].header["FLUXMAX"]
            """
            for item in [CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, fluxmax]:
                if type(item) == str: # Missing data gets inputted as a blank float - this catches it
                    raise CAIRDExceptions.CorruptImage("Missing metadata")
            """
            # Checks if the dimension of the images is correct in case the stamp was near the edge of the frame
            if stamp.shape != (50,50) or stamp is None or stamp.shape == 0:
                print(stamp.shape)
                raise CAIRDExceptions.CorruptImage("Dimension check failed")
                pass
            
            # NaN check
            if np.isnan(data).any() == True:
                raise CAIRDExceptions.CorruptImage("NaN data in stamp - skipping")
                pass
            
            # Stops you from going to hell
            if np.max(data) <= 0:  # Avoids dividing by zero; thank you Prof. Hafez!
                raise CAIRDExceptions.CorruptImage("Zero max in stamp - avoiding going to hell")
                pass
            
        else:
            hdu1[0].header["CLASSID"] = CID
            hdu1[0].header["DETECTIONID"] = TID
            hdu1[0].header["RA"] = RA
            hdu1[0].header["DEC"] = DEC
            hdu1[0].header["FLUXRAD"] = fluxrad
            hdu1[0].header["ELLIPTICITY"] = ellipticity
            hdu1[0].header["FWHM"] = fwhm
            hdu1[0].header["FLUXMAX"] = fluxmax
            """
            for item in [CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, fluxmax]:
                if type(item) == str: # Missing data gets inputted as a blank float - this catches it
                    raise CAIRDExceptions.CorruptImage("Missing metadata")
           """ 
            # Checks if the dimension of the images is correct in case the stamp was near the edge of the frame
            if stamp.shape != (50, 50) or stamp is None or stamp.shape == 0:
                print(stamp.shape)
                raise CAIRDExceptions.CorruptImage("Dimension check failed")
                pass
            
            # NaN check
            if np.isnan(data).any() == True:
                raise CAIRDExceptions.CorruptImage("NaN data in stamp - skipping")
                pass
            
            # Stops you from going to hell
            if np.max(data) <= 0:  # Avoids dividing by zero; thank you Prof. Hafez!
                raise CAIRDExceptions.CorruptImage("Zero max in stamp - avoiding going to hell")
                pass
        # Writes to new file
        if stamp.shape != (50, 50) or stamp is None or stamp.shape == 0: # Checks if the dimension of the images is correct in case the stamp was near the edge of the frame
            print(stamp.shape)
            raise CAIRDExceptions.CorruptImage("Dimension check failed")
            pass
        if save == True:
            fits.writeto(output, stamp, hdu1[0].header, overwrite=True)
        else:
            return stamp, [RA, DEC, fwhm, ellipticity, fluxrad, fluxmax]

"""

InputProcessor(scipath, refpath, diffpath, outputdir, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)

A wrapper function which takes in the 3 images, their metadata, the output directory, then returns the stacked image and the arrayed metadata.

"""


def InputProcessor(scipath, refpath, diffpath, DoStamp, outputdir, xpos, ypos, CID = None, TID = None, RA = None, DEC = None, fluxrad = None, ellipticity = None, fwhm = None, bkg = None, fluxmax = None):
    
    SciImg, SciMD = Stamper(scipath, os.path.join(outputdir, "ProcessorTemp.clean.fits"), False, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax, DoStamp = DoStamp)
    RefImg, RefMD = Stamper(refpath, os.path.join(outputdir, "ProcessorTemp.ref.fits"), False, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax, DoStamp = DoStamp)
    DiffImg, DiffMD = Stamper(diffpath, os.path.join(outputdir, "ProcessorTemp.diff.fits"), False, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax, DoStamp = DoStamp)
    
    StackedImg = np.stack((SciImg, RefImg, DiffImg), axis=2)
    
    return StackedImg, SciMD

def ImageGenerator(cleanpath):
    for scipath in cleanpath:
        refpath = scipath[:-10] + "ref.fits"
        diffpath = scipath[:-10] + "diff.fits"
        # Get label based upon directory
        if "/Bogus/" in scipath: 
            label = 0
        elif "/SN/" in scipath:
            label = 1
        else:
            continue
        try:
            StackedImg, StackMD = InputProcessor(scipath, refpath, diffpath, False, "", 25, 25)
            for n in range(1, 4):
                yield np.rot90(StackedImg, n), label
        except CAIRDExceptions.CorruptImage as Error:
            print(Error)
            continue

def DatasetGenerator():
    # Get all the files we need in a list
    print("Generating filepaths...")
    filepaths = glob.glob(os.path.join(DatabaseDir, "SortedData/**/*.clean.fits"), recursive=True)
    print(len(filepaths), "FP LENGTH")
    
    print("Shuffling filepaths...")
    random.seed(CAIRD.Randnum)
    random.shuffle(filepaths)
    print("Generating dataset...")
    dataset = tf.data.Dataset.from_generator(
        lambda: ImageGenerator(filepaths),
        output_signature=(
            tf.TensorSpec(shape=(50, 50, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
    dataset = dataset.batch(64).prefetch(buffer_size=tf.data.AUTOTUNE).shuffle(buffer_size = 256, seed = CAIRD.Randnum + CAIRD.Randnum)
    dataset.save(os.path.join(CAIRD.DatabaseDir, "TF/CurrentDataset"))
    print("Dataset saved!")
