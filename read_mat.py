#!/usr/bin/env python

"""process_mat.py:

__author__ = "William Bosl"
__copyright__ = "Copyright 2020, William J. Bosl"
__credits__ = ["William Bosl"]
__license__ = All rights reserved by William J. Bosl
__version__ = "1.0.0"
__maintainer__ = "William Bosl"
__email__ = "wjbosl@gmail.com"
__status__ = "Initial test"
"""

#
#  Copyright (c) 2020 William J. Bosl. All rights reserved.
#
import numpy as np
import os
import scipy.io as sio
import math

# A few global variables
f_labels = []
DEVICE = "unk"
AGE = 0
FORMAT = "long"
SEGMENT = "beg"  # "beg" = beginning, "mid" = middle, "end" = end. Position from which to extract the segment
SUBSAMPLE = False

# Use these to convert EGI net numbers to standard 10-20 location terminology
# The last entries for each here are not standard, but we've included them for completeness
hydroCell128_channelList = [22,    9,   33,  24,  11, 124, 122,  45,  36, 129, 104, 108,  58,  52,  62,  92,  96,  70,  83,   6]
egi_channel_names =       ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2","Fcz"]
EGI_64v2 =                [ 11,    6,   15,  13,  3,   62,  61,  24,  17,  65,  54,  52,  27,  28,  34,  46,  49,  37,  40,   4]
#EGI_64v1 =                [ 10,    5,   18,  12,  6,   60,  58,  24,  20,  65,  50,  52,  30,  28,  34,  42,  44,  35,  39,   8]

# the master list is just used for filtering all the extraneous channels that are typically saved in the Epilepsy Center files
master_channel_list = ["Fp1","Fp2","FP1","FP2","T3","T5","T4","T6","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2"]
processed_channel_names = []

#----------------------------------------------------------------------
# Convert matlab struct to a Python dict
#----------------------------------------------------------------------
def struct_to_dict(struct, name=None):
    result = dict()
    try:
        vals = struct[0,0]
    except IndexError:
        #print name, struct
        vals = ''
    try:
        for name in struct.dtype.names:
            if vals[name].shape == (1,):
                result[name] = vals[name][0]
            else:
                result[name] = struct_to_dict(vals[name], name=name)
    except TypeError:
        return vals
    return result


#----------------------------------------------------------------------
# Extract the time series and sensor names from the file. We will
# process only .edf and .csv files at this time.
#----------------------------------------------------------------------
def get_data(fullpathname, max_nt, params):

    file = open(fullpathname,'r')
    last = len(fullpathname)

    if fullpathname[last-4] == '.':
        tag = fullpathname[last-3:last]
    elif fullpathname[last-5] == '.':
        tag = fullpathname[last-4:last]
    else:
        print ("Cannot read tag. Exiting.")
        exit()

    resample = False # default value
    if "resample" in params:
        if params["resample"] == '0':
            resample = False
        else:
            resampling_rate = float(params["resample"])
            resample = True

    PROCESS_FILE = False
    if tag.lower() == "mat":
        channelNames = egi_channel_names
        filename = os.path.basename(fullpathname)
        ID = filename[:-4]
        #ID = ntpath.basename(filename)[:-4]

        # ---- matlab v7.3 or greater
        #print("Read h5")
        #mat_contents = h5py.File(filename, 'r')
        #print("keys:")
        #for k in mat_contents.keys():
        #    print(k)
        #eeg3 = mat_contents['EEG3']
        #data = mat_contents['EEG3']
        #print("size of EEG3: ", eeg3.keys())
        #exit()
        #--------------------------------

        mat_contents = sio.loadmat(fullpathname)
        #mat_proc_info = struct_to_dict(mat_contents['file_proc_info'])
        #data = mat_contents['Category_1_Segment1']
        keys = list(mat_contents.keys())
        if ID not in keys:
            ID = ID + '_'  # this is a funky mistake in a few files we need to process
        
        try:
            data = mat_contents[ID]
        except:
            print ("ID:", ID)
            print ("mat_contents keys:")
            print(mat_contents.keys())
            exit()

        nch, nt = data.shape
        srate = mat_contents['samplingRate'][0,0]
        
        # Here we assume that the desired sampling rate is an integer multiple of srate
        nt2 = int(math.ceil(nt/2))
        new_data = np.zeros((nch, nt2))
        if srate > resampling_rate:
            skip = int(srate/resampling_rate)
            print("Original / new sampling rate = ", srate, resampling_rate)
            srate = resampling_rate
            for ch in range(nch):
                resamp = data[ch][0:nt:skip]
                new_data[ch] = resamp
        data = new_data
        
        print("data shape = ", nch, nt)
        print("srate = ", srate)
        PROCESS_FILE = True

    if PROCESS_FILE:
        data_channels, nt = data.shape

        # We generally want only the standard 10-20 channels (19 max)
        # The high density EGI nets have numbered channel names. We'll need to convert.
        new_data = []
        new_channel_list = []

        # These two sections are for extracting desired channels from high density EGI devices
        # You can ignore this if data from any other devices are being processes.
        if len(data) == 129:  # EGI hydrocell 128
            for c, ch in enumerate(egi_channel_names):
                new_channel_list.append(ch)
                i = hydroCell128_channelList[c] - 1
                new_data.append(data[i])
            DEVICE = "EGI hydrocell 128"
            channelNames = new_channel_list
            data = new_data

        elif len(data) == 65:  # EGI 64 v2
            for c, ch in enumerate(egi_channel_names):
                new_channel_list.append(ch)
                i = EGI_64v2[c] - 1
                new_data.append(data[i])
            DEVICE = "EGI 64 v2"
            channelNames = new_channel_list
            data = new_data

        # Let's trim the data array so that time series are not more than max_nt seconds
        max_nt_points = max_nt * srate
        nt = int(min(max_nt_points, nt))

        # Here we're just picking out a segment of max_nt seconds from the entire EEG time series.
        # Start at the beginning (beg), middle (mid), or end of the entire array.
        m = len(data)
        n = len(data[0])
        if "beg" in SEGMENT:
            m1 = 0
        if "end" in SEGMENT:
            m1 = n-nt
        elif "mid" in SEGMENT:
            m1 = int((n-nt)/2)
        else:
            m1 = 0
        m2 = m1 + nt
        new_data = []
        for i in range(m):
            new_data.append( np.array(data[i][m1:m2]) )

        data = np.array(new_data)

    # Return a numpy array with the selected data, the channel names, and the sampling rate.
    return data, channelNames, srate


#
#----------------------------------------------------------------------
# Main
#----------------------------------------------------------------------
if __name__ == "__main__":
    print ("This file must be called from elsewhere.")


