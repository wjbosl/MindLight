#!/usr/bin/env python

"""compute_RP.py:

__author__ = "William Bosl"
__copyright__ = "Copyright 2020, William J. Bosl"
__credits__ = ["William Bosl"]
__license__ = All rights reserved by William J. Bosl
__version__ = "1.0.0"
__maintainer__ = "William Bosl"
__email__ = "william.bosl@childrens.harvard.edu"
__status__ = "Initial test"
"""
#
#  Copyright (c) 2020 William J. Bosl. All rights reserved.
#

import sys
import os
import numpy as np

# pyunicorn
import pyunicorn.timeseries.recurrence_plot as unirp

# A few global variables
CZ_REF = False
WRITE_RP_IMAGE_FILE = True

f_labels = []

# the master list is just used for filtering all the extraneous channels that are typically saved in the Epilepsy Center files
# EGI high density arrays are numbered 1-65 or 1-129. For other systems, we'll assume the standard array designations.
master_channel_list = list(range(1,129)) + ["AF3", "AF4", "FC5", "FC6", "Fp1", "Fp2", "FP1", "FP2", "T3", "T5", "T4", "T6", "F7", "F3", "Fz", "F4", "F8", "T7", "C3",
                       "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]

#master_channel_list = list(["F7"])

#master_channel_list = ['CC1', 'CC2', 'CC3', 'CC4', 'CC5', 'CC6', 'CC7', 'CC8', 'CC9', 'CC10', 
#'CC11', 'CC12', 'CC13', 'CC14', 'CC15', 'CC16', 'CC17', 'CC18', 'CC19', 'CC20', 
#'CC21', 'CC22', 'CC23', 'CC24', 'CC25', 'CC26', 'CC27', 'CC28', 'CC29', 'CC30', 
#'CC31', 'CC32', 'CC33', 'CC34', 'CC35', 'CC36', 'CC37', 'CC38', 'CC39', 'CC40', 
#'CC41', 'CC42', 'CC43', 'CC44', 'CC45', 'CC46', 'CC47', 'CC48', 'CC49', 'CC50', 
#'CC51', 'CC52', 'CC53', 'CC54', 'CC55', 'CC56', 'CC57', 'CC58', 'CC59', 'CC60', 
#'CC61', 'CC62', 'CC63', 'CC64']

# ----------------------------------------------------------------------
# Features to be computed for each scale from Recurrence networks:
#   Scales are relative to the sampling rate of the digital time
#   series.
# ----------------------------------------------------------------------
def get_RP(data, channelNames, srate, params=None):
    global f_labels, processed_channel_names
    
    # Default RQA parameters
    embedding = 10  # Embedding dimension
    tdelay = 2  # Time delay
    tau = 30  # threshold

    D = {}
    processed_channel_names = []

    #channelNames = ["Fp1","Fp2"] # for testing only
    for c, ch in enumerate(channelNames):
        if ch in master_channel_list:
            processed_channel_names.append(ch)

            D[ch] = {}
            # Check data quality: bad channels typically have zero power.

	    # Subsample if the sampling rate is high.
            if srate == 1024:
                n = len(data[c])
                y = data[c][0:n:2]
            else:
                y = data[c]
            v = np.sum(y**2)/len(y)

            # --------------------------------------------------------------------
            # Compute features on each of the frequency bands
            # --------------------------------------------------------------------
            if v != 0.0:
                # ----------------------
                # Feature set 3: Recurrence Quantitative Analysis (RQA)

                # pyunicorn
                py_uni_rp = unirp.RecurrencePlot(y, dim=embedding, tau=tdelay, recurrence_rate=0.20, metric='euclidean',silence_level=2)
                rp_matrix = np.array(py_uni_rp.recurrence_matrix(), dtype='byte') # this is for saving the RP image
                print("Completed channel ", ch)
                D[ch] = rp_matrix
            else:
                n = len(y)
                D[ch] = np.zeros((n,n))
                     
    return D


