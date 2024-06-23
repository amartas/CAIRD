#!/usr/bin/env python
# Author: Aidan Martas

import os
import CAIRD


DatabaseDir = CAIRD.DatabaseDir
MLDir = CAIRD.MLDir





#CAIRD.DatasetPreparation()

CAIRD.BuildCAIRD(MLDir, DatabaseDir)