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



    
reg_compile = re.compile(".*_NoGoal_.*")
subset_dirs = np.array([dirname for dirname in dirnames if  reg_compile.match(dirname)])
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
runs = [r[16:] for r in subset_dirs]
plt.legend(runs)
plt.ylabel("Mean return")
plt.xlabel("Env observations")
# plt.savefig(str(tits[1]) + '2.png', dpi=300)
# plt.savefig(str(tits[1]) + '2.pdf')  

