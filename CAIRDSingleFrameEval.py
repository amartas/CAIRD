# Author: Aidan Martas
import CAIRD
from astropy.io import fits

scipath = "/home/entropian/Documents/DLT40/FinalCAIRDDatasets/Artifacts/1.clean.fits"
refpath = "/home/entropian/Documents/DLT40/FinalCAIRDDatasets/Artifacts/1.ref.fits"
diffpath = "/home/entropian/Documents/DLT40/FinalCAIRDDatasets/Artifacts/1.diff.fits"


with fits.open(scipath) as hdu1:
    CID = hdu1[0].header["CLASSID"]
    TID = hdu1[0].header["DETECTIONID"]
    RA = hdu1[0].header["RA"]
    DEC = hdu1[0].header["DEC"]
    fluxrad = hdu1[0].header["FLUXRAD"]
    ellipticity = hdu1[0].header["ELLIPTICITY"]
    fwhm = hdu1[0].header["FWHM"]
    bkg = hdu1[0].header["BKG"]
    fluxmax = hdu1[0].header["FLUXMAX"]



print(CAIRD.ClassifyImage(scipath, refpath, diffpath, 25, 25, CID, TID, RA, DEC, fluxrad, ellipticity, fwhm, bkg, fluxmax))
