#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:53:44 2020

@author: arjun
"""
import numpy as np
import Energy_Feature_Extraction as en
import MFCC_Feature_Extraction as mfcc
import Pitch_Feature_Extraction as ptch
import ZCR_Feature_Extraction as ZCR
import os.path

emotions = (['Angry', 'Fearful', 'Happy', 'Monotone', 'Neutral', 'Sad'])
feature_matrix = np.zeros((144,1))
emotion_label = ["start"]
speech_info = ["start"]

for i in range(1,2):
    for idx, word in enumerate(emotions):
        for j in range(1,5):
            filepath = '/home/arjun/Downloads/VESUS/Audio/' + str(i) + '/' + word + '/' + str(j) +'.wav'
            if (os.path.exists(filepath) == False):
                continue
            
            energy = en.energy_features(filepath)
            MFCC = mfcc.mfcc_features(filepath)
            pitch = ptch.pitch_features(filepath)
            zcr = ZCR.zcr_features(filepath)
            features = np.concatenate((energy, MFCC, pitch, zcr), axis=0)
            print(pitch)
            feature_matrix = np.concatenate((feature_matrix, features), axis=1)
            for k in range(len(MFCC[0])):
                emotion_label.append(word)
                speech_info.append(str(i) + " " + word + str(j))
