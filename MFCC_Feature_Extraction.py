# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import librosa
import numpy as np

def mfcc_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    frames = librosa.util.frame(y, frame_length=17640, hop_length=8820, axis=0)

    mean = np.zeros((13, len(frames[:,0])))
    maxim = np.zeros((13, len(frames[:,0])))
    minim = np.zeros((13, len(frames[:,0])))
    variance = np.zeros((13, len(frames[:,0])))
    diff_mean = np.zeros((13, len(frames[:,0])))
    diff_max = np.zeros((13, len(frames[:,0])))
    diff_min = np.zeros((13, len(frames[:,0])))
    diff_var = np.zeros((13, len(frames[:,0])))


    for i in range(len(frames[:,0])):
        MFCC = librosa.feature.mfcc(y=frames[i], sr=44100,n_mfcc=13, hop_length=1102, win_length=1102)
        
        mean[:,i] = np.mean(MFCC,axis=1)
        maxim[:,i] = np.max(MFCC,axis=1)
        minim[:,i] = np.min(MFCC,axis=1)
        variance[:,i] = np.var(MFCC,axis=1)
        
        MFCC_zero_padded = np.concatenate((np.zeros((13,1)),MFCC),axis=1)
        MFCC_first_diff = np.diff(MFCC_zero_padded)
        diff_mean[:,i] = np.mean(MFCC_first_diff, axis=1)
        diff_max[:,i] = np.max(MFCC_first_diff, axis=1)
        diff_min[:,i] = np.min(MFCC_first_diff, axis=1)
        diff_var[:,i] = np.var(MFCC_first_diff, axis=1)
        
    MFCC_features = np.concatenate((mean, maxim, minim, variance, diff_mean, diff_max, diff_min, diff_var), axis=0)
    
    return MFCC_features    
    
