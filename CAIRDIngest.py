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
import time
import CAIRDExceptions
import CAIRD

# If you wish to make an apple pie from scratch, you must first invent the universe.  - Carl Sagan

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

def Stamper(img, output, save, x_pos = 25, y_pos = 25, CID = None, TID = None, RA = None, DEC = None, fluxrad = None, ellipticity = None, fwhm = None, bkg = None, fluxmax = None):
    
    with fits.open(img) as hdu1:
        # Extracts the data
        data = hdu1[0].data
        y, x = data.shape
        stamp = data[int(y_pos) - 11 : int(y_pos) + 11, int(x_pos) - 11 : int(x_pos) + 11] # Crops the image
        
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
        else:
            hdu1[0].header["CLASSID"] = CID
            hdu1[0].header["DETECTIONID"] = TID
            hdu1[0].header["RA"] = RA
            hdu1[0].header["DEC"] = DEC
            hdu1[0].header["FLUXRAD"] = fluxrad
            hdu1[0].header["ELLIPTICITY"] = ellipticity
            hdu1[0].header["FWHM"] = fwhm
            hdu1[0].header["FLUXMAX"] = fluxmax
            
            UniformityCheck(stamp, 2)
            
        if stamp.shape != (22,22) or stamp is None or stamp.shape == 0: # Checks if the dimension of the images is correct in case the stamp was near the edge of the frame
            print(stamp.shape)
            raise CAIRDExceptions.CorruptImage("Pre-padding dimension check failed")
            pass
        else:
            # Writes to new file
            stamp = np.pad(stamp, 3, mode = "constant", constant_values = 0)
            if stamp.shape != (28,28) or stamp is None or stamp.shape == 0: # Checks if the dimension of the images is correct in case the stamp was near the edge of the frame
                print(stamp.shape)
                raise CAIRDExceptions.CorruptImage("Post-padding dimension check failed")
                pass
            if save == True:
                fits.writeto(output, stamp, hdu1[0].header, overwrite=True)
            else:
                return stamp, [RA, DEC, fwhm, ellipticity, fluxrad, fluxmax]

"""

ImageStacker(directory, dim, badcount, imgname)

Loads, checks, and stacks 3 input images into a 3D NumPy array; loads, checks, and stacks metadata into another array, both of which can be fed to CAIRD for classification.

directory - Directory in which the images-to-be-stacked are located
dim       - Dimension of the data (nearly always 51)
imgname   - Image NAME, no extension, just the image name (e.g. 242, not 242.clean.fits or /SN/242).  Expects all images to be of the form [integer].*.fits ascending from zero.

"""

def ImageStacker(directory, dim, imgname):
    IsBad = False
    StackedImg = None
    MetadataInst = np.empty(6)
    # Set up paths
    try:
        SciencePath = os.path.join(directory, f"{str(imgname)}.clean.fits")
        RefPath = os.path.join(directory, f"{str(imgname)}.ref.fits")
        DiffPath = os.path.join(directory, f"{str(imgname)}.diff.fits")
        print(imgname)
    except FileNotFoundError:
        IsBad = True
        print(imgname, "does not exist")
        pass
    # Blinded by the light
    with fits.open(SciencePath) as hdu1:
        ScienceImg = hdu1[0].data
        
        # Dimension check
        if ScienceImg.shape != (dim, dim) or ScienceImg is None or ScienceImg.shape == 0:
            print("Dimension check failed")
            IsBad = True
            pass
        
        # NaN check
        elif np.isnan(hdu1[0].data).any() == True:
            print("NaN data!")
            pass
        
        # Stops you from going to hell
        elif np.max(ScienceImg) <= 0:  # Avoids dividing by zero; thank you Prof. Hafez!
            print(str(imgname) + " has a zero max!  Avoiding going to hell...")
            IsBad = True
            pass
        
        # Continue
        else:
            ScienceImg = ScienceImg - np.min(ScienceImg)  # Normalize
            ScienceImg = ScienceImg / np.max(ScienceImg)
            # The dark evening stars
            with fits.open(RefPath) as hdu2:
                RefImg = hdu2[0].data
                
                # Dimension check
                if RefImg.shape != (dim, dim) or RefImg is None or RefImg.shape == 0:
                    print("Dimension check failed")
                    IsBad = True
                    pass
                
                # NaN check
                elif np.isnan(hdu2[0].data).any() == True:
                    print("NaN data!")
                    IsBad = True
                    pass
                
                # Stops you from going to hell
                elif np.max(RefImg) <= 0:
                    print(str(imgname) + " has a zero max!  Avoiding going to hell...")
                    IsBad = True
                    pass
                
                # Continue
                else:
                    RefImg = RefImg - np.min(RefImg)  # Normalize
                    RefImg = RefImg / np.max(RefImg)
                    # And the morning sky of blue
                    with fits.open(DiffPath) as hdu3:
                        DiffImg = hdu3[0].data
                        
                        # Dimension check
                        if DiffImg.shape != (dim, dim) or DiffImg is None or DiffImg.shape == 0:
                            print("Dimension check failed")
                            IsBad = True
                            pass
                        
                        # NaN check
                        elif np.isnan(hdu3[0].data).any() == True:
                            print("NaN data!")
                            IsBad = True
                            pass
                        
                        # Stops you from going to hell
                        elif np.max(DiffImg) <= 0:
                            print(
                                str(imgname) + " has a zero max!  Avoiding going to hell...")
                            IsBad = True
                            pass
                        
                        # Actually add the data
                        else:
                            DiffImg = DiffImg - np.min(DiffImg)  # Normalize
                            DiffImg = DiffImg / np.max(DiffImg)
                            
                            # Add metadata to another array
                            hdr = hdu1[0].header
                            count = 0
                            try:
                                for item in ["RA", "DEC", "FWHM", "ELLIPTICITY", "FLUXRAD", "FLUXMAX"]:
                                    MetadataInst[count] = hdr[item]
                                    #print("Added " + item, MetadataInst[count])
                                    count += 1
                                # Stack the images into a 3-channel array of 2D arrays
                                StackedImg = np.stack((ScienceImg, RefImg, DiffImg), axis=2)
                                print(StackedImg.shape)
                            except ValueError:
                                IsBad = True
                                # And I sent it in my letter to you
                                print(item + " is corrupted - skipping")
                                pass
    return StackedImg, MetadataInst, IsBad

"""

DatasetPreparation(dim)

Prepares the training data for ML training.

IMPORTANT: THIS NUMPY ARRAY CAN BECOME VERY LARGE.  BE PREPARED FOR HIGH MEMORY USAGE AND THE POSSIBLE NEED FOR MEMORY MANAGEMENT!

"""

def DatasetPreparation(dim):  # Prepares the numpy arrays for ML training

    def DSRotator(arr, outputdir):
        
        def rotate_image(img, n):
            return tf.image.rot90(img, n)
        print(arr.shape)
        DS_0 = rotate_image(arr, 0)
        DS_0 = tf.transpose(DS_0, [0, 1, 2, 3])
        
        np.save(os.path.join(outputdir, "Temp_0.npy"), DS_0)
        print(DS_0.shape)
        del DS_0
        
        DS_90 = rotate_image(arr, 1)
        DS_90 = tf.transpose(DS_90, [0, 1, 2, 3])
        
        np.save(os.path.join(outputdir, "Temp_90.npy"), DS_90)
        print(DS_90.shape)
        del DS_90
        
        DS_180 = rotate_image(arr, 2)
        DS_180 = tf.transpose(DS_180, [0, 1, 2, 3])
        
        np.save(os.path.join(outputdir, "Temp_180.npy"), DS_180)
        print(DS_180.shape)
        del DS_180
        
        DS_270 = rotate_image(arr, 3)
        DS_270 = tf.transpose(DS_270, [0, 1, 2, 3])
        
        np.save(os.path.join(outputdir, "Temp_270.npy"), DS_270)
        print(DS_270.shape)
        del DS_270

        return np.concatenate((np.load(os.path.join(outputdir,"Temp_0.npy")), np.load(os.path.join(outputdir,"Temp_90.npy")), np.load(os.path.join(outputdir,"Temp_180.npy")), np.load(os.path.join(outputdir,"Temp_270.npy"))))

    def DatasetStacker(dim):
        count = 0
        
        
        for scipath in glob(os.path.join(DatabaseDir, "SortedData/Bogus/*/*.clean.fits")):
            refpath = scipath[:-10] + "ref.fits"
            diffpath = scipath[:-10] + "diff.fits"
            try:
                Stamper(scipath, os.path.join(DatabaseDir, "ProcessedData/Bogus/" + str(count) + ".clean.fits"), True)
                Stamper(refpath, os.path.join(DatabaseDir, "ProcessedData/Bogus/" + str(count) + ".ref.fits"), True)
                Stamper(diffpath, os.path.join(DatabaseDir, "ProcessedData/Bogus/" + str(count) + ".diff.fits"), True)
                count += 1
            except (CAIRDExceptions.CorruptImage, FileNotFoundError) as Error:
                print(Error)
                pass
        count = 0
        for scipath in glob(os.path.join(DatabaseDir, "SortedData/SN/*.clean.fits")):
            refpath = scipath[:-10] + "ref.fits"
            diffpath = scipath[:-10] + "diff.fits"
            try:
                Stamper(scipath, os.path.join(DatabaseDir, "ProcessedData/SN/" + str(count) + ".clean.fits"), True)
                Stamper(refpath, os.path.join(DatabaseDir, "ProcessedData/SN/" + str(count) + ".ref.fits"), True)
                Stamper(diffpath, os.path.join(DatabaseDir, "ProcessedData/SN/" + str(count) + ".diff.fits"), True)
                count += 1
            except (CAIRDExceptions.CorruptImage, FileNotFoundError) as Error:
                print(Error)
                pass
        """
        count = 0
        for scipath in glob(os.path.join(DatabaseDir, "SortedData/VarStars/*.clean.fits")):
            refpath = scipath[:-10] + "ref.fits"
            diffpath = scipath[:-10] + "diff.fits"
            try:
                Stamper(scipath, os.path.join(DatabaseDir, "ProcessedData/VarStars/" + str(count) + ".clean.fits"))
                Stamper(refpath, os.path.join(DatabaseDir, "ProcessedData/VarStars/" + str(count) + ".ref.fits"))
                Stamper(diffpath, os.path.join(DatabaseDir, "ProcessedData/VarStars/" + str(count) + ".diff.fits"))
                count += 1
            except (CAIRDExceptions.CorruptImage, FileNotFoundError) as Error:
                print(Error)
                pass
        
        """
        imgcount = sum([len(os.listdir(os.path.join(DatabaseDir, "ProcessedData/Bogus/"))),
                          len(os.listdir(os.path.join(DatabaseDir, "ProcessedData/SN/")))
                          ])
        ImgDatabase = np.empty((imgcount//3 * 4, dim, dim, 3))
        ImgLabels = np.empty((imgcount//3 * 4))
        ImgMetadata = np.empty((imgcount//3 * 4, 6))
        print(ImgDatabase.shape)
        
        count = 0
        badcount = 0
        for scipath in glob(os.path.join(DatabaseDir, "ProcessedData/Bogus/*.clean.fits")):
            print(scipath)
            StackedImg, MetadataInst, IsBad = ImageStacker(os.path.join(DatabaseDir, "ProcessedData/Bogus/"), dim, os.path.basename(scipath)[:-11])
            print(os.path.basename(scipath)[:-11], "Base Name")
            
            
            if IsBad == True:
                print("Bad image at", badcount)
                badcount += 1
                pass
            
            else:
                ImgDatabase[count] = StackedImg
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 0
                print("Processed bogus image", count)
                print(ImgDatabase.shape, "SHAPE")
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 1)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 0
                print("Processed bogus image", count)
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 2)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 0
                print("Processed bogus image", count)
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 3)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 0
                print("Processed bogus image", count)
                count += 1
                
        for scipath in glob(os.path.join(DatabaseDir, "ProcessedData/SN/*.clean.fits")):
            print(scipath)
            StackedImg, MetadataInst, IsBad = ImageStacker(os.path.join(DatabaseDir, "ProcessedData/SN/"), dim, os.path.basename(scipath)[:-11])
            
            if IsBad == True:
                print("Bad image at", badcount)
                badcount += 1
                pass
            
            else:
                ImgDatabase[count] = StackedImg
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 1
                print("Processed SN image", count)
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 1)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 1
                print("Processed SN image", count)
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 2)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 1
                print("Processed SN image", count)
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 3)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 1
                print("Processed SN image", count)
                count += 1
        """
        for scipath in glob(os.path.join(DatabaseDir, "ProcessedData/VarStars/*.clean.fits")):
            print(scipath)
            StackedImg, MetadataInst, IsBad = ImageStacker(os.path.join(DatabaseDir, "ProcessedData/VarStars/"), dim, os.path.basename(scipath)[:-11])
            
            if IsBad == True:
                print("Bad image at", badcount)
                badcount += 1
                pass
            
            else:
                ImgDatabase[count] = StackedImg
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 2
                print("Processed varstar image", count)
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 1)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 2
                print("Processed varstar image", count)
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 2)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 2
                print("Processed varstar image", count)
                count += 1
                ImgDatabase[count] = np.rot90(StackedImg, 3)
                ImgMetadata[count] = MetadataInst
                ImgLabels[count] = 2
                print("Processed varstar image", count)
                count += 1    
        """
        ImgDatabase = ImgDatabase[:(-4 * badcount - 1)]
        ImgMetadata = ImgMetadata[:(-4 * badcount - 1)] 
        ImgLabels = ImgLabels[:(-4 * badcount - 1)]
        
        print(badcount, "Bad image count")


        print(ImgDatabase.shape, ImgLabels.shape, ImgMetadata.shape, "Shapes")
        
        # Holdout construction
        
        ImgDatabase, ImgLabels, ImgMetadata = shuffle(ImgDatabase, ImgLabels, ImgMetadata, random_state=CAIRD.Randnum)
        print(ImgDatabase.shape, ImgLabels.shape, ImgMetadata.shape)
        
        HoldoutSize = int(len(ImgDatabase) * 0.05)
        print(HoldoutSize)
        
        ImgDatabase, Holdout1Imgs = ImgDatabase[:(-1 * HoldoutSize)], ImgDatabase[(-1 * HoldoutSize):]
        ImgLabels, Holdout1Labels = ImgLabels[:(-1 * HoldoutSize)], ImgLabels[(-1 * HoldoutSize):]
        ImgMetadata, Holdout1Metadata = ImgMetadata[:(-1 * HoldoutSize)], ImgMetadata[(-1 * HoldoutSize):]
        
        print(ImgDatabase.shape, ImgLabels.shape, ImgMetadata.shape)
        
        ImgDatabase, Holdout2Imgs = ImgDatabase[:(-1 * HoldoutSize)], ImgDatabase[(-1 * HoldoutSize):]  # You'd better have a damn good reason for touching this holdout
        ImgLabels, Holdout2Labels = ImgLabels[:(-1 * HoldoutSize)], ImgLabels[(-1 * HoldoutSize):]
        ImgMetadata, Holdout2Metadata = ImgMetadata[:(-1 * HoldoutSize)], ImgMetadata[(-1 * HoldoutSize):]
        
        MMImgDatabase = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/ImgDatabase.memmap"), dtype = ImgDatabase.dtype, mode = "w+", shape = ImgDatabase.shape)
        MMImgLabels = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/ImgLabels.memmap"), dtype = ImgLabels.dtype, mode = "w+", shape = ImgLabels.shape)
        MMImgMetadata = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/ImgMetadata.memmap"), dtype = ImgMetadata.dtype, mode = "w+", shape = ImgMetadata.shape)
        
        MMImgDatabase[:] = ImgDatabase[:]
        MMImgDatabase.flush()
        MMImgLabels[:] = ImgLabels[:]
        MMImgLabels.flush()
        MMImgMetadata[:] = ImgMetadata[:]
        MMImgMetadata.flush()
        
        MMHoldout1Imgs = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/Holdout1Imgs.memmap"), dtype = Holdout1Imgs.dtype, mode = "w+", shape = Holdout1Imgs.shape)
        MMHoldout1Labels = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/Holdout1Labels.memmap"), dtype = Holdout1Labels.dtype, mode = "w+", shape = Holdout1Labels.shape)
        MMHoldout1Metadata = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/Holdout1Metadata.memmap"), dtype = Holdout1Metadata.dtype, mode = "w+", shape = Holdout1Metadata.shape)
        
        MMHoldout1Imgs[:] = Holdout1Imgs[:]
        MMHoldout1Imgs.flush()
        MMHoldout1Labels[:] = Holdout1Labels[:]
        MMHoldout1Labels.flush()
        MMHoldout1Metadata[:] = Holdout1Metadata[:]
        MMHoldout1Metadata.flush()
        
        MMHoldout2Imgs = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/Holdout2Imgs.memmap"), dtype = Holdout2Imgs.dtype, mode = "w+", shape = Holdout2Imgs.shape)
        MMHoldout2Labels = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/Holdout2Labels.memmap"), dtype = Holdout2Labels.dtype, mode = "w+", shape = Holdout2Labels.shape)
        MMHoldout2Metadata = np.memmap(os.path.join(CAIRD.DatabaseDir, "Arrays/Holdout2Metadata.memmap"), dtype = Holdout2Metadata.dtype, mode = "w+", shape = Holdout2Metadata.shape)
        
        print(ImgDatabase.dtype)
        
        ArrInfo =  {
                   "ImgDatabase": ImgDatabase.shape,
                   "ImgLabels": ImgLabels.shape,
                   "ImgMetadata": ImgMetadata.shape,
                   "Holdout1Imgs": Holdout1Imgs.shape,
                   "Holdout1Labels": Holdout1Labels.shape,
                   "Holdout1Metadata": Holdout1Metadata.shape,
                   "Holdout2Imgs": Holdout2Imgs.shape,
                   "Holdout2Labels": Holdout2Labels.shape,
                   "Holdout2Metadata": Holdout2Metadata.shape
                   }
        
        DatabaseInfo = open(os.path.join(CAIRD.DatabaseDir, "Arrays/DatasetInfo.dict"), "w")
        DatabaseInfo.write(str(ArrInfo))
        DatabaseInfo.close()
        
        
        
        MMHoldout2Imgs[:] = Holdout2Imgs[:]
        MMHoldout2Imgs.flush()
        MMHoldout2Labels[:] = Holdout2Labels[:]
        MMHoldout2Labels.flush()
        MMHoldout2Metadata[:] = Holdout2Metadata[:]
        MMHoldout2Metadata.flush()
        
        print(ImgDatabase.shape, ImgLabels.shape, ImgMetadata.shape)
        
        del ImgDatabase, ImgLabels, ImgMetadata # Save memory
        
        print("Finished dataset creation")
    
    DatasetStacker(28)
    
    
"""

InputProcessor(scipath, refpath, diffpath, outputdir, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)

A wrapper function which takes in the 3 images, their metadata, the output directory, then returns the stacked image and the arrayed metadata.

"""


def InputProcessor(scipath, refpath, diffpath, outputdir, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax):
    
    SciImg, SciMD = Stamper(scipath, os.path.join(outputdir, "ProcessorTemp.clean.fits"), False, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)
    RefImg, RefMD = Stamper(refpath, os.path.join(outputdir, "ProcessorTemp.ref.fits"), False, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)
    DiffImg, DiffMD = Stamper(diffpath, os.path.join(outputdir, "ProcessorTemp.diff.fits"), False, xpos, ypos, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax)
    
    StackedImg = np.stack((SciImg, RefImg, DiffImg), axis=2)
    
    return StackedImg, SciMD
    










































































