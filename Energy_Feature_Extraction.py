#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:22:08 2020

@author: arjun
"""
import librosa
import numpy as np
from scipy.stats import kurtosis, skew

def energy_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    frames = librosa.util.frame(y, frame_length=17640, hop_length=8820, axis=0)
    
    energy = np.zeros((1,4))
    
    frame_energy = np.zeros((1,len(frames[:,0])))
    mean = np.zeros((1, len(frames[:,0])))
    variance = np.zeros((1, len(frames[:,0])))
    maxim = np.zeros((1, len(frames[:,0])))
    rnge = np.zeros((1, len(frames[:,0])))
    kurt = np.zeros((1, len(frames[:,0])))
    skw = np.zeros((1, len(frames[:,0])))
    diff_mean = np.zeros((1, len(frames[:,0])))
    diff_variance = np.zeros((1, len(frames[:,0])))
    diff_max = np.zeros((1, len(frames[:,0])))
    diff_range = np.zeros((1, len(frames[:,0])))
    diff_kurt = np.zeros((1, len(frames[:,0])))
    diff_skew = np.zeros((1, len(frames[:,0])))
    
    for i in range(len(frames[:,0])):
        frames_roll = librosa.util.frame(frames[i], frame_length=4410, hop_length=4410, axis=0)
        for j in range(len(frames_roll[:,0])):
            fft = np.fft.fft(frames_roll[j])
            energy[:,j] = np.sum(np.square(fft))
                
            frame_energy[:,i] = np.sum(energy)
            mean[:,i] = np.mean(energy)
            variance[:,i]  = np.var(energy)
            maxim[:,i]  = np.max(energy)
            rnge[:,i]  = maxim[:,i] - np.min(energy)
            kurt[:,i]  = kurtosis(energy,axis=1)
            skw[:,i]  = skew(energy,axis=1)
                
            energy_padded = np.concatenate((np.zeros((1,1)),energy),axis=1)
            diff_energy = np.diff(energy_padded)
            diff_mean[:,i]  = np.mean(diff_energy)
            diff_variance[:,i]  = np.var(diff_energy)
            diff_max[:,i]  = np.max(diff_energy)
            diff_range[:,i]  = diff_max[:,i] - np.min(diff_energy)
            diff_kurt[:,i]  = kurtosis(diff_energy,axis=1)
            diff_skew[:,i]  = skew(diff_energy,axis=1)
                
    energy_features = np.concatenate((frame_energy, mean, variance, maxim, rnge, kurt, skw, diff_mean, diff_variance, diff_max, diff_range, diff_kurt, diff_skew), axis=0)
     
    return energy_features       
    
