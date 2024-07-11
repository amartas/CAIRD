# Author: Aidan Martas
import CAIRD
import argparse
"""
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", type=str, required=True, help="Full filepath for the science image")
parser.add_argument("--xpos", type=float, required=True, help="X-position of the detection")
parser.add_argument("--ypos", type=float, required=True, help="Y-position of the detection")
parser.add_argument("--classid", type=int, required=False, help="Classification ID of the detection")
parser.add_argument("--targetid", type=int, required=False, help="Target ID of the detection")
parser.add_argument("--ra", type=float, required=False, help="Right Ascension of the detection")
parser.add_argument("--dec", type=float, required=False, help="Declination of the detection")
parser.add_argument("--fluxrad", type=float, required=False, help="Fluxrad of the detection")
parser.add_argument("--ellipticity", type=float, required=False, help="Ellipticity of the detection")
parser.add_argument("--fwhm", type=float, required=False, help="FWHM of the detection")
parser.add_argument("--bkg", type=float, required=False, help="Background of the image")
parser.add_argument("--fluxmax", type=float, required=False, help="Fluxmax of the image")

args = parser.parse_args()

scipath = args.file
refpath = scipath[:-10] + "ref.fits"
diffpath = scipath[:-10] + "diff.fits"

Classification = CAIRD.ClassifyImage(scipath, 
                                     refpath, 
                                     diffpath, 
                                     args.xpos, 
                                     args.ypos, 
                                     args.classid, 
                                     args.targetid, 
                                     args.ra, 
                                     args.dec, 
                                     args.fluxrad, 
                                     args.ellipticity, 
                                     args.fwhm, 
                                     args.bkg, 
                                     args.fluxmax
                                     )


print(Classification)
"""

num = 14
imgtype = "SN"


cleanpath = "/home/entropian/Documents/CAIRD/CAIRDDatasets/SortedData/" + str(imgtype) + "/" + str(num) + ".clean.fits"
refpath = "/home/entropian/Documents/CAIRD/CAIRDDatasets/SortedData/" + str(imgtype) + "/" + str(num) + ".ref.fits"
diffpath = "/home/entropian/Documents/CAIRD/CAIRDDatasets/SortedData/" + str(imgtype) + "/" + str(num) + ".diff.fits"

print(CAIRD.ClassifyImage(cleanpath, refpath, diffpath, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0))