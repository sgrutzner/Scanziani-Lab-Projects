#walking rythem_vdrive #cleaned

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import scipy.signal as sig
import ghostipy as gsp
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, norm
from tqdm import tqdm
import matplotlib

# Setting matplotlib parameters
matplotlib.rcParams.update({'font.size': 22})

#PARAMS to alter to fit each animal

#this is in the index of ttls not a time
#rec_end_in_ttl_time = 1795025 #smgvl6_230418 first session
####rec_end_in_ttl_time = 1919254#mlr3 first day first session
rec_end_in_ttl_time = 1712478 #mlr3 second day first session

step = 0.005#seconds

##############################################
# Function to load data
# if I want to use this I have to convert end_time from ttl to rec_time and then 
#change this function to have data that is < rec_time
# def load_data(file_path, end_time=None):
#     data = np.load(file_path, allow_pickle=True)
#     if end_time is not None:
#         data = data[:end_time]
#     return data

# load dict of spike times in seconds
spike_dict = np.load('data/' + 'spike_dict.npy', allow_pickle=True) 

# load DLC lables (one label per frame)
fl_paw_speed = np.load('data/' + 'front_left_paw_speed.npy')
fr_paw_speed = np.load('data/' + 'front_right_paw_speed.npy')
bl_paw_speed = np.load('data/' + 'rear_left_paw_speed.npy')
br_paw_speed = np.load('data/' + 'rear_right_paw_speed.npy')

# load camera TTL times (seconds) corresponding to each frame
cam_ttl_times = np.load('data/' + 'camera_ttl_times.npy') 

#if you want to combine sessions change this section
cam_ttl_times = cam_ttl_times[:rec_end_in_ttl_time]
fl_paw_speed = fl_paw_speed[:rec_end_in_ttl_time]
fr_paw_speed = fr_paw_speed[:rec_end_in_ttl_time]
bl_paw_speed = bl_paw_speed[:rec_end_in_ttl_time]
br_paw_speed = br_paw_speed[:rec_end_in_ttl_time]

#change this to full rec vs single video more simply 
normalized_time = np.arange(np.min(cam_ttl_times), np.max(cam_ttl_times), step)#np.linspace(dlc_time[int(lift_times[ee])]- window_s, dlc_time[int(lift_times[ee])]+window_s, num =  199)#
#normalized_time = np.arange(np.min(cam_ttl_times), np.max(np.hstack(spike_dict)), step)#np.linspace(dlc_time[int(lift_times[ee])]- window_s, dlc_time[int(lift_times[ee])]+window_s, num =  199)#

binned_spike_dict = []
for ii in tqdm(range(0, np.size(spike_dict))):
    binned_spike_dict.append( np.histogram(spike_dict[ii], bins = normalized_time)[0])

end_time = cam_ttl_times[-1]
start_time = cam_ttl_times[0]
binned_spike_dict = np.array(binned_spike_dict)[:,:int(end_time/step)]

smooth_spike_dict = []
for ii in tqdm(range(0, np.shape(binned_spike_dict)[0])):
    kernel = np.ones(8)
    smooth_spike_dict.append(scipy.signal.fftconvolve(binned_spike_dict[ii], kernel, mode='same'))

##########################################################################
##########################################################################

cam_ttl_intervals = np.diff(cam_ttl_times)

# use the mean interval between camera TTLs to get sampling rate
print('Mean interval between camera TTLs (s): {}'.format(np.mean(cam_ttl_intervals)))
fs_paw_speed = 1/np.mean(cam_ttl_intervals)
print('True average sampling rate of paw speed files: {} Hz'.format(fs_paw_speed))

# make it an integer for convenience 
fs_paw_speed = int(np.round(fs_paw_speed))
print('Rounding to an int for convenience: {} Hz'.format(fs_paw_speed))

# how much data is there?
print('Recording length (mins): {}'.format(len(fl_paw_speed)/fs_paw_speed/60))

##########################################################################
##########################################################################


# Function to get spectral content via continuous wavelet transform

def apply_cwt(data, fs, gamma=3, beta=100, freq_lims=[0.5, 20]):
    """
    Apply Continuous Wavelet Transform (CWT) to the given data.

    Parameters:
    data (numpy.ndarray): The data to be transformed.
    fs (float): Sampling frequency.
    gamma (int): Gamma parameter for CWT.
    beta (int): Beta parameter for CWT.
    freq_lims (list): Frequency limits for CWT.

    Returns:
    numpy.ndarray: The CWT coefficients.
    numpy.ndarray: The frequencies corresponding to the CWT coefficients.
    """
    n_samps = len(data)
    T = n_samps / fs
    t = np.linspace(0, T, n_samps, endpoint=False)

    cwtcoefs, _, freq, _, _ = gsp.cwt(
        data, timestamps=t, freq_limits=freq_lims, fs=fs,
        wavelet=gsp.MorseWavelet(gamma=gamma, beta=beta))
    cwt = cwtcoefs.imag**2 + cwtcoefs.real**2
    return cwt / np.max(cwt), freq

# Parameters
gamma = 3
beta = 100
freq_lims = [0.5, 20]
paws = [fl_paw_speed, fr_paw_speed, bl_paw_speed, br_paw_speed]
labels = ['front left', 'front right', 'back left', 'back right']

# Paw frequency lists
fl_paw_freq, fr_paw_freq, bl_paw_freq, br_paw_freq = [], [], [], []

# Apply CWT to each paw speed data
for i, data in enumerate(tqdm(paws)):
    cwt, freq = apply_cwt(data, fs_paw_speed, gamma, beta, freq_lims)
    if i == 0:
        fl_paw_freq.append(cwt)
    elif i == 1:
        fr_paw_freq.append(cwt)
    elif i == 2:
        bl_paw_freq.append(cwt)
    elif i == 3:
        br_paw_freq.append(cwt)

    # Optional: Plot spectrogram
    # fig, ax = plt.subplots()
    # fig.set_size_inches((12,3)) 
    # t_ax, f_ax = np.meshgrid(np.linspace(0, len(data) / fs_paw_speed, len(data)), freq)
    # ax.pcolormesh(t_ax, f_ax, cwt, shading='gouraud', cmap=plt.cm.viridis, vmin=0, vmax=0.6)
    # ax.set_ylabel("Frequency (Hz)")
    # ax.set_xlabel("Time (s)")
    # ax.set_title(labels[i])
    # plt.show()

fl_paw_freq = np.vstack(fl_paw_freq)
fr_paw_freq = np.vstack(fr_paw_freq)
bl_paw_freq = np.vstack(bl_paw_freq)
br_paw_freq = np.vstack(br_paw_freq)

relevant_freq_idx = np.where(np.logical_and(freq>=4, freq<=7))[0]
avg_freq = np.mean([fl_paw_freq[relevant_freq_idx], fr_paw_freq[relevant_freq_idx], bl_paw_freq[relevant_freq_idx],br_paw_freq[relevant_freq_idx]], axis = 0)
avg_freq = np.mean(avg_freq, axis = 0)

still_freq_idx = np.where(freq>=3)[0]
still_freq = np.mean([fl_paw_freq[still_freq_idx], fr_paw_freq[still_freq_idx], bl_paw_freq[still_freq_idx],br_paw_freq[still_freq_idx]], axis = 0)
still_freq = np.mean(still_freq, axis = 0)

########################################################################

#these may have to be adjusted/changed based on the animal
#walk_threshold = [0.0007, 10] 
#still_threshold = 0.000008
#for mlr day2
walk_threshold = [0.02, 10] #these may have to be adjusted/changed based on the animal
still_threshold = 0.000015#

window_size = int(fs_paw_speed/6)#
slide = int(fs_paw_speed/8)#
behave = []
avg_val = []

def find_behave(avg_freq, window_size, walk_threshold, still_threshold):
    """
    Identify behavior based on average frequency and thresholds.
    Parameters:
    avg_freq (numpy.ndarray): Array of average frequencies.
    still_freq (numpy.ndarray): Array of frequencies for still behavior.
    window_size (int): Size of the sliding window.
    slide (int): Slide step for the window.
    walk_threshold (list): Threshold for walking behavior.
    still_threshold (float): Threshold for still behavior.
    Returns:
    tuple: Arrays of behavior and average values.
    """
    num_windows = int(len(avg_freq) / window_size * (window_size/slide))
    for i in range(num_windows):
        start = i*slide
        window = avg_freq[start:start+window_size]
        still_pow = still_freq[start:start+window_size]
        if np.mean(window) > walk_threshold[0]:#, np.mean(window) < walk_threshold[1]):
            if np.all(window)< 15:
                behave.append(1)#walk
        elif np.mean(still_pow) < still_threshold:
            behave.append(-1)#still
        else: 
            behave.append(0)
        avg_val.append(np.mean(window))
    return np.hstack(behave), np.hstack(avg_val) #half second bins slid by a fourth of a second 1 = walk, -1 = still, 0 = too noisy to determine

find_behave(avg_freq, window_size, walk_threshold, still_threshold)
##############################################################################
xaxis1 = np.linspace(start_time,cam_ttl_times[-1], np.size(fl_paw_speed))
xaxis2 = np.linspace(start_time,cam_ttl_times[-1], np.size(behave))

#now check that the thresholds are accuratly capturing the running
#and adjust as needed
plt.plot(xaxis2,behave)
plt.plot(xaxis1,avg_freq*10)
plt.plot(xaxis1, fl_paw_speed, color = 'grey', alpha = 0.5)
plt.show()

