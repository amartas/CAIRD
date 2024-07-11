#!/usr/bin/env python
# Author: Aidan Martas

import os
import CAIRD
import CAIRDIngest

DatabaseDir = CAIRD.DatabaseDir
MLDir = CAIRD.MLDir


# Ingest training data, preprocess it, and then build a new CAIRD model
#CAIRDIngest.DatasetGenerator()
CAIRD.BuildCAIRD(MLDir, DatabaseDir)