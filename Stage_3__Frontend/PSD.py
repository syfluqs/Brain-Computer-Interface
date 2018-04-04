import statistics
import numpy as np
from scipy import signal
from scipy.stats import kurtosis

n = 100  # number of elements in feature-vector

# Extracting PSD features
def PSD_extractor(x):
    pxx, f  = signal.periodogram(x)
    f = f[0:-1]/1e6
    return_vector = np.zeros([3*n])
    step = int(len(f)/n)
    for i in range(n):
        avg = np.mean(f[step*i:step*(i+1)])
        stn_dev = statistics.stdev(f[step*i:step*(i+1)]) 
        kurto = kurtosis(f[step*i:step*(i+1)])
        return_vector[i] = avg
        return_vector[n+i] = stn_dev
        return_vector[2*n+i] = kurto
    return return_vector