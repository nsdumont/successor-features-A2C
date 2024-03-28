import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from hyperparam_opt import optim
import utils
import numpy as np

env = 'maze-sample-5x5-v0'
n_frames = 15000
algos = np.array(['ppo'])
wrappers = np.array(['xy'])#, 'ssp'])
n_seeds = 3
n_trials=100
params = {}
for algo in algos:
    for wrap in wrappers:
        if wrap=='xy':
            inwrap='none'
            dissim_coef= 0
            lr=0.0001
            entropy_coef=0.002
            entrpoy_decay=0.01
            batch_size=256
            epochs=5
        if 'ssp' in wrap:
            dissim_coef =0.0#float(wrap.split('_')[1])
            inwrap='ssp-auto'
            lr=0.001
            entropy_coef=0.0002
            entrpoy_decay=0.01
            batch_size=64
            epochs=1
            
        final_param = optim(env,algo,inwrap,'flat',n_frames, n_seeds,n_trials, 
            initial_params={
                "gae_lambda": 0.8,
                "procs": 4,
                "entropy_coef": entropy_coef,
                "entrpoy_decay": entrpoy_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "ssp_h": 1.0,
                "feature_size": 128,
                "dissim_coef": 0.0,
                "actor_hidden_size": 64,
                "clip_eps": 0.2,
                "critic_hidden_size": 64,
                "discount": 0.99
                }
            )
        params[algo + '-' + wrap] = final_param
 
        
        
            

                
            
        
            

