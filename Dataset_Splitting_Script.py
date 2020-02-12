#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:14:41 2020

@author: arjun
"""
import scipy.io as scio
import numpy as np
dataset = scio.loadmat('/home/arjun/.config/spyder-py3/full_features.mat')
features = dataset["features"]
labels = dataset["labels"]
info = dataset["info"]
zeros_label = np.zeros(0)
ones_label = np.zeros(0)
twos_label = np.zeros(0)
threes_label = np.zeros(0)
fours_label = np.zeros(0)
fives_label = np.zeros(0)

for i in range(len(features[:,0])):
    if labels[i]==0:
        zeros_label = np.append(zeros_label,[i])
    if labels[i]==1:
        ones_label = np.append(ones_label,[i])
    if labels[i]==2:
        twos_label = np.append(twos_label,[i])
    if labels[i]==3:
        threes_label = np.append(threes_label,[i])
    if labels[i]==4:
        fours_label = np.append(fours_label,[i])
    if labels[i]==5:
        fives_label = np.append(fives_label,[i])

zeros_label = zeros_label.astype(int)
ones_label = ones_label.astype(int)
twos_label = twos_label.astype(int)
threes_label = threes_label.astype(int)
fours_label = fours_label.astype(int)
fives_label = fives_label.astype(int)
zeros_features = features[zeros_label,:]
ones_features = features[ones_label,:]
twos_features = features[twos_label,:]
threes_features = features[threes_label,:]
fours_features = features[fours_label,:]
fives_features = features[fives_label,:]
zeros_info = info[zeros_label]
ones_info = info[ones_label]
twos_info = info[twos_label]
threes_info = info[threes_label]
fours_info = info[fours_label]
fives_info = info[fives_label]
scio.savemat('zero_features.mat', {"features":zeros_features, "labels":np.zeros(zeros_label.shape[0]).astype(int), "info":zeros_info})
scio.savemat('one_features.mat', {"features":ones_features, "labels":np.ones(ones_label.shape[0]).astype(int), "info":ones_info})
scio.savemat('two_features.mat', {"features":twos_features, "labels":np.full(twos_label.shape[0], 2, dtype=int), "info":twos_info})
scio.savemat('three_features.mat', {"features":threes_features, "labels":np.full(threes_label.shape[0], 3, dtype=int), "info":threes_info})
scio.savemat('four_features.mat', {"features":fours_features, "labels":np.full(fours_label.shape[0], 4, dtype=int), "info":fours_info})
scio.savemat('five_features.mat', {"features":fives_features, "labels":np.full(fives_label.shape[0], 5, dtype=int), "info":fives_info})
        