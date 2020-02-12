#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:09:24 2020

@author: arjun
"""

import librosa
import numpy as np
from scipy.stats import kurtosis, skew

def zcr_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    frames = librosa.util.frame(y, frame_length=17640, hop_length=8820, axis=0)
    
    mean = np.zeros((1, len(frames[:,0])))
    variance = np.zeros((1, len(frames[:,0])))
    maxim = np.zeros((1, len(frames[:,0])))
    rnge = np.zeros((1, len(frames[:,0])))
    kurt = np.zeros((1, len(frames[:,0])))
    skw = np.zeros((1, len(frames[:,0])))
    diff_mean = np.zeros((1, len(frames[:,0])))
    diff_variance = np.zeros((1, len(frames[:,0])))
    diff_max = np.zeros((1, len(frames[:,0])))
    diff_min = np.zeros((1, len(frames[:,0])))
    diff_range = np.zeros((1, len(frames[:,0])))
    diff_kurt = np.zeros((1, len(frames[:,0])))
    diff_skew = np.zeros((1, len(frames[:,0])))
    
    for i in range(len(frames[:,0])):
        zero_cross = librosa.feature.zero_crossing_rate(frames[i], frame_length=4410, hop_length=4410)
        
        mean[:,i] = np.mean(zero_cross)
        variance[:,i]  = np.var(zero_cross)
        maxim[:,i]  = np.max(zero_cross)
        rnge[:,i]  = maxim[:,i] - np.min(zero_cross)
        kurt[:,i]  = kurtosis(zero_cross,axis=1)
        skw[:,i]  = skew(zero_cross,axis=1)
        
        zero_cross_padded = np.concatenate((np.zeros((1,1)),zero_cross),axis=1)
        diff_zero_cross = np.diff(zero_cross_padded)
        diff_mean[:,i]  = np.mean(diff_zero_cross)
        diff_variance[:,i]  = np.var(diff_zero_cross)
        diff_max[:,i]  = np.max(diff_zero_cross)
        diff_min[:,i]  = np.min(diff_zero_cross)
        diff_range[:,i]  = diff_max[:,i] - diff_min[:,i]
        diff_kurt[:,i]  = kurtosis(diff_zero_cross,axis=1)
        diff_skew[:,i]  = skew(diff_zero_cross,axis=1)
    
    ZCR_features = np.concatenate((mean, variance, maxim, rnge, kurt, skw, diff_mean, diff_variance, diff_max, diff_min, diff_range, diff_kurt, diff_skew), axis=0) 
    
    return ZCR_features