#!/usr/bin/env python

"""recur_net.py:

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
import numpy as np
import scipy
from scipy import signal

# For wavelet decomposition into frequency bands
import pywt
RECONSTRUCT = True
APPROXIMATIONS = True

# For nonlinear values
import nolds

# For synchronization
COMPUTE_SYNC = False
if COMPUTE_SYNC:
    import compute_sync as sync

# pyunicorn
pyUnicorn = True
if pyUnicorn:
    import pyunicorn.timeseries.recurrence_plot as unirp

# For RQA
pyRQA = False
if pyRQA:
    from pyrqa.settings import Settings
    from pyrqa.neighbourhood import FixedRadius, RadiusCorridor
    from pyrqa.metric import EuclideanMetric
    from pyrqa.computation import RQAComputation
    from pyrqa.computation import RPComputation
    from pyrqa.image_generator import ImageGenerator
    from pyrqa.time_series import TimeSeries

# A few global variables
CZ_REF = False
FORMAT = "long"
WRITE_RP_IMAGE_FILE = False

#all_features = ["Power", "SampE", "RR", "DET", "LAM", "Lentr", "Lmax", "TT", "MeanRT",'VertEnt', 'WhiteVertEnt','AvgWhiteVertLen']
#all_features = ["Power", "SampE", "lyap0", "dfa", "cd", "RR", "DET", "LAM", "Lentr", "Lmean", "Lmax", "TT", 'VertEnt', 'AvgWhiteVertLen', "MeanRT", 'WhiteVertEnt']
all_features = ["Power", "SampE", "cd", "dfa","lyap0","lyap1","lyap2", "RR", "DET", "LAM", "Lentr", "Lmean", "Lmax", "TT", 'VertEnt', 'AvgWhiteVertLen']
#all_features = ["Power", "SampE","lyap0", "RR", "DET", "LAM", "Lentr", "Lmean", "Lmax", "TT", 'VertEnt', 'AvgWhiteVertLen']
#all_features = ["Power", "SampE"]
#all_features = ["Power", "SampE", "RR", "DET", "LAM", "DIV", "Lentr", "Lmax", "Lmean", "TT"]
f_labels = []

# the master list is just used for filtering all the extraneous channels that are typically saved in the Epilepsy Center files
# EGI high density arrays are numbered 1-65 or 1-129. For other systems, we'll assume the standard array designations.
master_channel_list = list(range(1, 129)) + ["AF3", "AF4", "FC5", "FC6", "Fp1", "Fp2", "FP1", "FP2", "T3", "T5", "T4",
                                             "T6", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                                             "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2", "E1", "E2"]

#master_channel_list = ['EEG F3-C3', 'EEG C3-P3', 'EEG P3-O1', 'EEG F4-C4', 'EEG C4-P4',
#                       'EEG P4-O2', 'EEG T3 -T5', 'EEG T5-T6', 'EEG C4-M1','EEG O2-M1']


#master_channel_list = list(["F7"])

# master_channel_list = ['CC1', 'CC2', 'CC3', 'CC4', 'CC5', 'CC6', 'CC7', 'CC8', 'CC9', 'CC10',
# 'CC11', 'CC12', 'CC13', 'CC14', 'CC15', 'CC16', 'CC17', 'CC18', 'CC19', 'CC20',
# 'CC21', 'CC22', 'CC23', 'CC24', 'CC25', 'CC26', 'CC27', 'CC28', 'CC29', 'CC30',
# 'CC31', 'CC32', 'CC33', 'CC34', 'CC35', 'CC36', 'CC37', 'CC38', 'CC39', 'CC40',
# 'CC41', 'CC42', 'CC43', 'CC44', 'CC45', 'CC46', 'CC47', 'CC48', 'CC49', 'CC50',
# 'CC51', 'CC52', 'CC53', 'CC54', 'CC55', 'CC56', 'CC57', 'CC58', 'CC59', 'CC60',
# 'CC61', 'CC62', 'CC63', 'CC64']

processed_channel_names = []


# ----------------------------------------------------------------------
# Total power
# From: https://raphaelvallat.com/bandpower.html
# "In order to compute the average bandpower in the delta band, we first
# need to compute an estimate of the power spectral density. The most
# widely-used method to do that is the Welch's periodogram, which consists
# in averaging consecutive Fourier transform of small windows of the signal,
# with or without overlapping.

#   The Welch's method improves the accuracy of the classic periodogram.""
# ----------------------------------------------------------------------
def bandpower(x, srate, high, low):
    f, Pxx = signal.periodogram(x, fs=srate)
    ind_min = scipy.argmax(f > low) - 1
    ind_max = scipy.argmax(f > high) - 1
    avg_power = (scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max]))/1.0e9

    return avg_power


# ----------------------------------------------------------------------
# Features to be computed for each scale from Recurrence networks:
#   Scales are relative to the sampling rate of the digital time
#   series.
# ----------------------------------------------------------------------
def get_signal_features(data, channelNames, srate, params=None):
    global f_labels, processed_channel_names

    # Default RQA parameters
    embedding = 20  # Embedding dimension
    tdelay = 2  # Time delay
    tau = 0.00001  # threshold
    RRconstant = 0.05  # Constant recurrence rate
    
    # Input parameters
    if params == None:
        write_to_screen = False
    else:
        write_to_screen = params["write_to_screen"]
    print("write_to_screen = ", write_to_screen)
        
    # Multiscaling is accomplished with a wavelet transform
    # Options for basis functions: ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey']
    # wavelet = 'haar'
    wavelet = 'db4'
    mode = 'cpd'
    # mode = pywt.Modes.smooth

    # Determine the number of levels required so that
    # the lowest level approximation is roughly the
    # delta band (freq range 0-4 Hz)

    if srate <= 128:
        levels = 5
    elif srate <= 256:
        levels = 6
    elif srate <= 512:
        levels = 7
    elif srate <= 1024:
        levels = 8
    else:
        levels = 9
    nbands = levels

    # The following function returns the highest level (ns) approximation
    # in dec[0], then details for level ns in dec[1]. Each successive
    # level of detail coefficients is in dec[2] through dec[ns].
    #
    #   level       approximation       details
    #   0           original signal     --
    #   1                -              dec[ns]
    #   2                -              dec[ns-1]
    #   3                -              dec[ns-2]
    #   i              -                dec[ns-i+1]
    #   ns          dec[0]              dec[1]

    # Print screen headers
    # sys.stdout.write("%10s %7s  %7s " % ("Sensor", "Freq", "Wavelet"))
    # for f in all_features:
    #    sys.stdout.write(" %8s " % (f))
    # sys.stdout.write("\n")

    D = {}
    y_ch_freq = {}

    # For Cz re-referencing
    if CZ_REF:
        c = channelNames.index('Cz')
        cz_ref = data[c]

    wavelet_scale = {}
    f_limit = {}

    print("Compute sub-series for all freq bands")
    print("\n")

    bad_channels = {}
    for c, ch in enumerate(channelNames):
        if ( ch in master_channel_list or ch.startswith("EEG") or ch.startswith("ECG") ):
        #if 1==1:
            if ch.startswith("EEG"):
                ch = ch[4:]
            processed_channel_names.append(ch)
            
            # --------------------------------------------------------------------
            # First we'll do a simple check for bad channels: if power of 
            # the original signal is zero, or Nan, then it's a bad channel. 
            # No need for further processing.
            bad_channels[ch] = False    # innocent until proven guilty
            y = data[c]
            avg_power = np.sum(y * y) / len(y)
            if avg_power==0.0 or avg_power==np.NaN:
                bad_channels[ch] = False   # guilty
             
            # For Baylor data only: rereference to Cz
            ts = data[c]
            if CZ_REF:
                ts = data[c] - cz_ref

            y_ch_freq[ch] = {}

            # --------------------------------------------------------------------
            # Get the wavelet decomposition. See pywavelet (or pywt) documents.
            # Deconstruct the waveforms
            # S = An + Dn + Dn-1 + ... + D1
            # --------------------------------------------------------------------
            w = pywt.Wavelet(wavelet)
            m = np.mean(ts)
            a_orig = ts - m  # the original signal, normalized to zero-mean
            a = a_orig

            ca = []  # all the approximations
            cd = []  # all the details
            s_ca = []  # all the approximations
            s_cd = []  # all the details
            sqrt2 = np.sqrt(2.0)
            scale_factor = sqrt2
            (a, d) = pywt.dwt(a, w, mode)
            ca.append(a)
            cd.append(d)
            s_ca.append(a / scale_factor)
            s_cd.append(d / scale_factor)
            for i in range(nbands):
                (a, d) = pywt.dwt(ca[i], w, mode)
                scale_factor = scale_factor * sqrt2
                ca.append(a)
                cd.append(d)
                s_ca.append(a / scale_factor)
                s_cd.append(d / scale_factor)

            if RECONSTRUCT:  # this will build full reconstructed signals at every level
                rec_a = []  # reconstructed approximations
                rec_d = []  # reconstructed details
                for i, coeff in enumerate(ca):
                    coeff_list = [coeff, None] + [None] * i
                    rec_a.append(pywt.waverec(coeff_list, w))
                for i, coeff in enumerate(cd):
                    coeff_list = [None, coeff] + [None] * i
                    rec_d.append(pywt.waverec(coeff_list, w))
            else:  # otherwise, use the power-of-2 smaller signals
                rec_a = s_ca
                rec_d = s_cd

            # Use the details and last approximation to create all the power-of-2 freq bands
            wavelet_scale = {}
            f_limit = {}
            freqband = []
            f_labels = []

            if APPROXIMATIONS:
                # First, A0, the original signal
                f_labels.append('A0')
                wavelet_scale['A0'] = 0
                f_limit['A0'] = srate / 2.0
                freqband.append(a_orig)  # A0 is the original signal
                N = len(a_orig)

                # Here we get all the approximations, A1, A2, ...
                # if 1==0:
                freq = srate / 4.0
                for j, r in enumerate(rec_a):
                    freq_name = 'A' + str(j + 1)
                    wavelet_scale[freq_name] = j + 1
                    f_limit[freq_name] = freq
                    freq = freq / 2.0
                    f_labels.append(freq_name)
                    freqband.append(r[0:N])  # wavelet approximation for this band

            # Here we create wavelet details, D1, D2, ...
            freq = srate / 2.0
            for j, r in enumerate(rec_d):
                freq_name = 'D' + str(j + 1)
                wavelet_scale[freq_name] = j + 1
                f_limit[freq_name] = freq
                freq = freq / 2.0
                f_labels.append(freq_name)
                freqband.append(r)  # wavelet details for this band
                
            # Save the last approximation band
            #j = len(rec_a)-1
            j = j
            freq_name = 'A' + str(j+1)
            wavelet_scale[freq_name] = j+1
            f_limit[freq_name] = freq
            f_labels.append(freq_name)
            freqband.append(rec_a[j])  # wavelet approximation for this band
            
            for i in range(len(f_labels)):
                y_ch_freq[ch][i] = freqband[i]

    #--------  end of loop to compute all sub-series using wavelet transform
            
    channelNames = processed_channel_names
    # if COMPUTE_SYNC is true, then we compute all synchronization features
    if COMPUTE_SYNC:
        for ch in channelNames:
            sync_name = 's_' + ch
            all_features.append(sync_name)
            
    for ch in channelNames:
        D[ch] = {}
        for feat in all_features:
            D[ch][feat] = {}
           
    # Screen headers, for monitoring progress during testing
    print("Compute properties for all channels: ", channelNames)
    print("\n")
    sys.stdout.write("%4s  " % ("Chan"))
    sys.stdout.write("%4s  " % ("Wave"))
    sys.stdout.write("%8s  " % ("Power"))
    for f in all_features:
        if f != "Power": sys.stdout.write("%8s  " % (f))
    sys.stdout.write("\n")

    for c, ch in enumerate(channelNames):
        
        # If it's a bad channel, assign NaN and go on to the next.
        if bad_channels[ch]:
            for feat in all_features:
                for w in f_labels:
                    D[ch][feat][w] = np.Nan
        
        else:    
        
            # --------------------------------------------------------------------
            # Compute features on each of the frequency bands
            # --------------------------------------------------------------------
            for i, f in enumerate(f_labels):
                y = y_ch_freq[ch][i]
                wavelet_level = f
                freq = f_limit[f]
    
                # ----------------------
                # Feature set 1: Power
                if "Power" in all_features:
                    avg_power = np.sum(y * y) / len(y)
                    #if avg_power < 1.0e-9:
                    avg_power *= 1.0e9
                    D[ch]["Power"][wavelet_level] = avg_power
    
                # ----------------------
                # Feature set 2: Sample Entropy, Hurst parameter, DFA, Lyapunov exponents
                if "SampE" in all_features:
                    try:
                        D[ch]["SampE"][wavelet_level] = nolds.sampen(y)
                    except:
                        D[ch]["SampE"][wavelet_level] = -999
    
                if "hurstrs" in all_features:
                    try:
                        D[ch]["hurstrs"][wavelet_level] = nolds.hurst_rs(y, fit='poly')
                    except:
                        D[ch]["hurstrs"][wavelet_level] = -999
    
                if "dfa" in all_features:
                    try:
                        D[ch]["dfa"][wavelet_level] = nolds.dfa(y)
                    except:
                        D[ch]["dfa"][wavelet_level] = -999
    
                if "cd" in all_features:
                    try:
                        if sys.version_info[0] >= 3:
                            D[ch]["cd"][f_labels[i]] = nolds.corr_dim(y, emb_dim=embedding, fit="poly")
                        else:
                            D[ch]["cd"][wavelet_level] = nolds.corr_dim(y, emb_dim=embedding, fit="poly")
                    except:
                        D[ch]["cd"][wavelet_level] = -999
    
                if "lyap0" in all_features:
                    try:
                        #lyap = nolds.lyap_e(y, emb_dim=embedding)  # use this to get several Lyap. Exponents
                        lyap = nolds.lyap_e(y)  # use this to get several Lyap. Exponents
                        lyap0 = lyap[0]
                        #lyap0 = nolds.lyap_r(y, emb_dim=embedding)
                        D[ch]["lyap0"][wavelet_level] = lyap0
                        if "lyap1" in all_features:
                            D[ch]["lyap1"][wavelet_level] = lyap[1]
                        if "lyap2" in all_features:
                            D[ch]["lyap2"][wavelet_level] = lyap[2]
                    except:
                        D[ch]["lyap0"][wavelet_level] = -999
    
                # ----------------------
                # Feature set 3: Recurrence Quantitative Analysis (RQA)
                # We have to compute the recurrence plot for any one of these variables.
                rqa_features = ["RR", "DET", "LAM", "L_entr", "L_max", "L_mean", "TT", "MeanRT", 'VertEnt',
                                'WhiteVertEnt', 'AvgWhiteVertLen']
                Compute_RQA = False
                for r in rqa_features:
                    if r in all_features:
                        Compute_RQA = True
                    break
    
                if Compute_RQA:
    
                    if pyRQA:  # This is for pyRQA routines. For now, we won't use this any longer. We use the pyUnicorn package instead.
                        time_series = TimeSeries(y,
                                                 embedding_dimension=embedding,
                                                 time_delay=tdelay)
    
                        settings = Settings(time_series,
                                            #computing_type=ComputingType.Classic,
                                            neighbourhood=FixedRadius(tau),
                                            #neighbourhood=RadiusCorridor(inner_radius=0.00001,outer_radius=0.00004),
                                            similarity_measure=EuclideanMetric
                                            # theiler_corrector=1,
                                            # min_diagonal_line_length=2,
                                            # min_vertical_line_length=2,
                                            # min_white_vertical_line_length=2)
                                            )
    
                        # This part of pyRQA remains the same regardless of version
                        computation = RQAComputation.create(settings, verbose=False)
                        result = computation.run()
    
                        # We have to pull out each value
                        if "RR" in all_features:
                            D[ch]["RR"][wavelet_level] = result.recurrence_rate
                        if "DET" in all_features:
                            D[ch]["DET"][wavelet_level] = result.determinism
                        if "LAM" in all_features:
                            D[ch]["LAM"][wavelet_level] = result.laminarity
                        if "DIV" in all_features:
                            D[ch]["DIV"][wavelet_level] = result.divergence
                        if "Lentr" in all_features:
                            D[ch]["Lentr"][wavelet_level] = result.entropy_diagonal_lines
                        if "Lmax" in all_features:
                            D[ch]["Lmax"][wavelet_level] = result.longest_diagonal_line
                        if "Lmean" in all_features:
                            D[ch]["Lmean"][wavelet_level] = result.average_diagonal_line
                        if "TT" in all_features:
                            D[ch]["TT"][wavelet_level] = result.trapping_time
    
                    if pyUnicorn:  # pyunicorn
                        py_uni_rp = unirp.RecurrencePlot(y, dim=embedding, tau=tdelay, recurrence_rate=RRconstant, metric='euclidean', silence_level=2)
                        #py_uni_rp = unirp.RecurrencePlot(y, dim=embedding, tau=tdelay, threshold=tau, metric='euclidean',silence_level=2)
                        # rp_matrix = py_uni_rp.recurrence_matrix() # this is for saving the RP image
    
                        # We have to pull out each value
                        if "RR" in all_features:
                            D[ch]["RR"][wavelet_level] = py_uni_rp.recurrence_rate()
                        if "DET" in all_features:
                            D[ch]["DET"][wavelet_level] = py_uni_rp.determinism()
                        if "LAM" in all_features:
                            D[ch]["LAM"][wavelet_level] = py_uni_rp.laminarity()
                        if "Lentr" in all_features:
                            D[ch]["Lentr"][wavelet_level] = py_uni_rp.diag_entropy()
                        if "Lmax" in all_features:
                            D[ch]["Lmax"][wavelet_level] = py_uni_rp.max_diaglength()
                        if "Lmean" in all_features:
                            D[ch]["Lmean"][wavelet_level] = py_uni_rp.average_diaglength()
                        if "TT" in all_features:
                            D[ch]["TT"][wavelet_level] = py_uni_rp.trapping_time()
                        if "MeanRT" in all_features:
                            D[ch]["MeanRT"][wavelet_level] = py_uni_rp.mean_recurrence_time()
                        if "VertEnt" in all_features:
                            D[ch]["VertEnt"][wavelet_level] = py_uni_rp.vert_entropy()
                        if "WhiteVertEnt" in all_features:
                            D[ch]["WhiteVertEnt"][wavelet_level] = py_uni_rp.white_vert_entropy()
                        if "AvgWhiteVertLen" in all_features:
                            D[ch]["AvgWhiteVertLen"][wavelet_level] = py_uni_rp.average_vertlength()
    
                # ----------------------
                # Feature set 4: Recurrence network values
                # TBD
                # Each of the variable names in rn_features should have a computed value here
                #
                if "rn_1" in all_features:
                    v = np.mean(y)
                    D[ch]["rn_1"][wavelet_level] = v
    
                # Compute the next variable
                if "rn_2" in all_features:
                    v = np.sqrt(np.abs(np.mean(y)))
                    D[ch]["rn_2"][wavelet_level] = v
    
                # ----------------------
                # Feature set 5: Global Field Power: GFP
                if "GFP" in all_features:
                    gfs = np.mean(y)  # Dummy filler for now
                    D[ch]["gfp"][wavelet_level] = v
                # TBD
    
                # ----------------------
                # Feature set 6: Global Field Synchronization: GFS
                if "GFS" in all_features:
                    gfs = np.mean(y)  # Dummy filler for now
                    D[ch]["gfs"][wavelet_level] = v
                # TBD
    
                # ----------------------
                # Feature set 7: Spectral Coherence
                # TBD
    
                # ----------------------
                # Feature set 8: Synchronization Likelihood
                # TBD
    
                # ----------------------
                # Feature set 9: Granger Causality
                # TBD
    
                # ----------------------
                # Feature set 10: Hilbert Phase Synchronization
                if COMPUTE_SYNC:
                    for ch2 in channelNames:
                        y2 = y_ch_freq[ch2][i]
                        sync_name = 's_' + ch2
                        D[ch][sync_name][wavelet_level] = sync.HilbertPhaseSync(y,y2)
                        #D[ch][sync_name][wavelet_level] = 0.5

                if write_to_screen:
                    sys.stdout.write("%4s  " % (ch))
                    sys.stdout.write("%4s  " % (wavelet_level))
                    for f in all_features:
                        sys.stdout.write("%8.3f  " % (D[ch][f][wavelet_level]))
                    sys.stdout.write("\n")
                # sys.stdout.write("\n")

    return D, srate, wavelet_scale, f_limit


