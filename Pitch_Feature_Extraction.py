#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:54:45 2020

@author: arjun
"""
import librosa
import numpy as np
from scipy.stats import kurtosis, skew
import pyworld as pw

def pitch_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    frames = librosa.util.frame(y, frame_length=17640, hop_length=8820, axis=0)

    mean = np.zeros((1, len(frames[:,0])))
    variance = np.zeros((1, len(frames[:,0])))
    maxim = np.zeros((1, len(frames[:,0])))
    minim = np.zeros((1, len(frames[:,0])))
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
        f0, t = pw.dio(frames[i].astype('double'), 44100, frame_period=50.0)
        f0 = np.reshape(f0,(1,9))
        mean[:,i] = np.mean(f0)
        variance[:,i] = np.var(f0)
        maxim[:,i] = np.max(f0)
        minim[:,i] = np.min(f0)
        rnge[:,i] = maxim[:,i] - np.min(f0)
        kurt[:,i] = kurtosis(f0,axis=1)
        skw[:,i] = skew(f0,axis=1)
        
        f0_padded = np.concatenate((np.zeros((1,1)),f0),axis=1)
        diff_f0 = np.diff(f0_padded)
        diff_mean[:,i] = np.mean(diff_f0)
        diff_variance[:,i] = np.var(diff_f0)
        diff_max[:,i] = np.max(diff_f0)
        diff_min[:,i] = np.min(diff_f0)
        diff_range[:,i] = diff_max[:,i] - diff_min[:,i]
        diff_kurt[:,i] = kurtosis(diff_f0,axis=1)
        diff_skew[:,i] = skew(diff_f0,axis=1)
        
    pitch_features = np.concatenate((mean, variance, maxim, minim, rnge, kurt, skw, diff_mean, diff_variance, diff_max, diff_min, diff_range, diff_kurt, diff_skew), axis=0)
    
    return pitch_features

