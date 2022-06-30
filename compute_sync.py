import numpy as np
import scipy
import scipy.fftpack as fftpack

def HilbertPhaseSync(y1,y2, windowSize=256, offset=0):
    n = len(y1)

    # Use a single sliding time window
    T = windowSize
    t0 = offset
    t1 = T + offset
    h = np.zeros(t1 - t0)

    # Compute either analytic phase (Hilbert transform) or Gaussian phase (convolution
    # with Gaussian kernel) for this window.
    h1 = scipy.arctan2(fftpack.hilbert(y1), y1)
    h2 = scipy.arctan2(fftpack.hilbert(y2), y2)
    diff = h1-h2
    
    gamma = abs(np.mean(np.exp(diff * 1.0j)))
    
    return(gamma)


def save_nm_sync():

    # Compute phase difference at each time, then phase index using
    # Shannon entropy  need only compute the upper half
    lagValues = [-30, -20, -10, 0, 10, 20, 30]
    #lagValues = [-240, -120, -40, 0, 12, 40, 120, 240]
    lagValues = [0]
    #nmPairs = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 3], [3, 4], [2, 1], [3, 1], [4, 1], [3, 2], [4, 3]]
    nmPairs = [[1, 1]]

    nlags = len(lagValues)
    Sarray = np.zeros((nlags * len(nmPairs)))

	# First, we compute surrogates and determine a baseline synchronization level
	# to overcome the systematic bias as discussed in Kreuz, "Measuring Synchronization
	# in coupled model systems: A comparison of different approaches", Physica D, 225, 2007.
	# Generate 10 random phase series and compute the synchronization index with these
	# to establish a baseline.

	# Shuffle one of the time series
    Smax = 0.0
    N = 10
    for i in range(N):
        h1_rand = scipy.arctan2(fftpack.hilbert(y1), y1)
        np.random.shuffle(h1_rand)
        diff = abs(h1_rand - h2)
        Smax += abs(np.mean(np.exp(diff * 1.0j)))
    Smax = Smax / N

	# For each n,m pair, evaluate different time lags, as synchronization could
	# change with the time delay since signals travel with finite time.
	# At 250 Hz, each step is 4 msec. We look at lags up to 120 msec in
	# 40 msec steps
    index = 0
    for lag in lagValues:
        s = y2[(t0+lag):(t1+lag)]
        h2 = scipy.arctan2(fftpack.hilbert(s), s)

		# Compute the maximum synchronization for a range of n,m values
        for i, p in enumerate(nmPairs):
            n = p[0]
            m = p[1]
            diff = abs(n * h1 - m * h2)
			# Phase index based on circular variance
            Sarray[index] = abs(np.mean(np.exp(diff * 1.0j)))
            index += 1

    g = np.max(Sarray)

    if g < Smax:
        gamma = 0.0
    else:
        gamma = (g - Smax) / (1.0 - Smax)

    return gamma
