#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:09:11 2020

@author: ns2dumon
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import os 
import re
Directory=os.getcwd() + '/storage'
dirnames = [name for name in os.listdir(Directory) if os.path.isdir(os.path.join(Directory, name))]


# Goal transfer task
regstrs = [".*_Goal_.*",".*_NoGoal_.*", ".*_size_.*"]
tits = ["Goal_transfer_learning", "Latent_learning","Size_transfer_learning"]
ridxs = [14,16,14]
for k in range(len(regstrs)):
    reg_compile = re.compile(regstrs[k])
    subset_dirs = [dirname for dirname in dirnames if  reg_compile.match(dirname)]
    column_name = 'return_mean'
    all_data = []
    frame_arrays=[] 
    plt.figure()
    for i in range(len(subset_dirs)):
        model_dir = subset_dirs[i]
        data = pd.read_csv(Directory + '/' + model_dir + "/log.csv")
        frame_arrays.append(pd.to_numeric(data['frames'], errors='coerce').values)
        all_data.append(pd.to_numeric(data[column_name], errors='coerce').values)
        plt.plot(frame_arrays[-1],all_data[-1],alpha=0.7,linewidth=1)
    runs = [r[ridxs[k]:] for r in subset_dirs]
    plt.legend(runs)
    plt.ylabel("Mean return")
    plt.xlabel("Env observations")
    plt.savefig(str(tits[k]) + '.png', dpi=300)
    plt.savefig(str(tits[k]) + '.pdf')
    plt.title(tits[k])    
    
    
reg_compile = re.compile(".*_NoGoal_.*")
subset_dirs = np.array([dirname for dirname in dirnames if  reg_compile.match(dirname)])
column_name = 'return_mean'
all_data = []
frame_arrays=[] 
plt.figure()
for i in range(len(subset_dirs)):
    model_dir = subset_dirs[i]
    if model_dir != "MiniGrid_NoGoal_SR_image":
        data = pd.read_csv(Directory + '/' + model_dir + "/log.csv")
        frame_arrays.append(pd.to_numeric(data['frames'], errors='coerce').values)
        all_data.append(pd.to_numeric(data[column_name], errors='coerce').values)
        plt.plot(frame_arrays[-1],all_data[-1],alpha=0.7,linewidth=1)
runs = [r[16:] for r in subset_dirs[[0,2,3]]]
plt.legend(runs)
plt.ylabel("Mean return")
plt.xlabel("Env observations")
plt.savefig(str(tits[1]) + '2.png', dpi=300)
plt.savefig(str(tits[1]) + '2.pdf')  

Directory=os.getcwd() + '/storage_oldi'
dirnames = [name for name in os.listdir(Directory) if os.path.isdir(os.path.join(Directory, name))]
reg_compile = re.compile(".*_Goal_.*")
subset_dirs = [dirname for dirname in dirnames if  reg_compile.match(dirname)]
column_name = 'return_mean'
all_data = []
frame_arrays=[] 
for i in range(len(subset_dirs)):
    model_dir = subset_dirs[i]
    data = pd.read_csv(Directory + '/' + model_dir + "/log.csv")
    frame_arrays.append(pd.to_numeric(data['frames'], errors='coerce').values)
    all_data.append(pd.to_numeric(data[column_name], errors='coerce').values)
runs = [r[14:] for r in subset_dirs]
runs = np.array(runs)

sub_rs = [[0,6],[3,5],[3,4],[2,6]]
for k in range(len(sub_rs)):
    plt.figure()
    for i in range(len(sub_rs[k])):
        frs = frame_arrays[sub_rs[k][i]]
        dat = all_data[sub_rs[k][i]]
        fid = frs <= 80000
        plt.plot(frs[fid],dat[fid],alpha=0.7,linewidth=1)
        plt.legend(runs[sub_rs[k]])
        plt.ylabel("Mean return")
        plt.xlabel("Env observations")
    plt.savefig('sr_params' + str(k) + '.png', dpi=300)
    plt.savefig('sr_params' + str(k) + '.pdf')
