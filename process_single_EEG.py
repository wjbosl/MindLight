#!/usr/bin/env python

"""process_single_EEG.py:

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

import sys, os
import read_edf
import read_mat
import signal_analysis
#import spectral_signal_analysis as signal_analysis
import write_to_csv
import compute_RP

# A few global variables
f_labels = []
DEVICE = "unk"
AGE = 0 # default value if age isn't known
FORMAT = "long"
SEGMENT = "beg"  # "beg" = beginning, "mid" = middle, "end" = end. Position from which to extract the segment
max_nt = 30
COMPUTE_FEATURES = True
COMPUTE_RP = False

#----------------------------------------------------------------------
# process a single file
#----------------------------------------------------------------------
def process(fullpathname, params):
    
    global AGE

    # Extract an ID from the file or filename
    filename = os.path.basename(fullpathname)
    n = len(filename)
    ID = filename[0:n-4]
    # For ISP files only
    isp_age = -1
    if filename[0:3] == "ISP":
        ID = filename[3:7]
        isp_age = int(filename[10:12])
        params["age"] = isp_age
    elif filename.endswith("YF.mat"):
        ID = filename[0:n-8]
        isp_age = int(filename[n-7])
        params["age"] = isp_age
    elif filename.endswith(".mat"):
        ID = filename[0:n-4]
        isp_age = 0
        params["age"] = isp_age
    tag = filename[n-3:]
    
    # Check if segment length was set
    if "max_nt" in params:
        max_nt = float(params["max_nt"])

    # for testing, we'll send messages about progress ...
    sys.stdout.write("Processing data file %s with tag %s \n" %(filename, tag) )

    npoints = 0
    if ((tag == "edf") or (tag=="EDF")):
        data, channelNames, srate = read_edf.get_data(fullpathname, max_nt, params)
        npoints = len(data[0])
        
    elif (tag == "mat"):
        data, channelNames, srate = read_mat.get_data(fullpathname, max_nt, params)
        npoints = len(data[0])

    elif (tag == "csv" or tag == "CSV"):
        data, channelNames, srate = read_csv.get_data(fullpathname, max_nt)
        npoints = len(data[0])

    else:
        print("file with tag .%s cannot be processed")
        npoints = -1

    # -----------------------------------
    # Now process the EEG data
    # -----------------------------------
    if (npoints < max_nt*srate and npoints > -1):
        print ("File ", filename, " does not have the required length and will not be processed.")
        return
    else:
        outfilename = params["outfilename"]
        fout = open(outfilename, 'a')
        
        # Loop over features and channels here
        if COMPUTE_FEATURES:
            D, srate, wavelet_scale, f_limit = signal_analysis.get_signal_features(data, channelNames, srate, params)
            write_to_csv.write_features(fout, D, ID, srate, f_limit, params)
            
        if COMPUTE_RP:
            D = compute_RP.get_RP(data, channelNames, srate)
            write_to_csv.write_RP(fout, D, ID, srate)

        fout.close()
        print ("File %s finished. Results written to %s." % (filename,fout.name))
        
    return

if __name__ == '__main__':
    
    #----------------------  Get input parameters arguments  ----------------------
    argv = sys.argv
    argc = len(argv)
    fullpathname = argv[1]
    outputfile = argv[2]
    
    # Default values
    params = {}
    params["outfilename"] = outputfile
    params["age"] = 0
    params["max_nt"] = 30
    params["SEGMENT"] = "mid"
    params["device"] = "dev"
    params["group"] = "group"
    params["resample"] = "0"
    params["write_to_screen"] = False
    
    # Read the input parameter file. Assume key, value pairs
    if argc == 4:
        param_file = open(argv[3], "r")
        lines = param_file.readlines()
        for i in range(len(lines)):
            key, value = lines[i].split()
            params[key] = value
    process(fullpathname, params)
        
