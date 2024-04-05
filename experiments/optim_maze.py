import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from hyperparam_opt import optim
import utils
import numpy as np


#PPO with xy
#Trial 83 finished with value: 0.8962450478690309 and parameters:
    #{'discount': 0.99, 'max_grad_norm': 0.8, 'gae_lambda': 0.95, 'lr': 0.0010650071035089798, 'entropy_coef': 0.000678150163515309, 'entropy_decay': 9.347513374190233e-07, 'procs': 6, 'value_loss_coef': 0.14648442322655858, 'actor_hidden_size': 128, 'critic_hidden_size': 64, 'feature_hidden_size': 128, 'clip_eps': 0.17566918588142932, 'batch_size': 32, 'epochs': 5}. Best is trial 83 with value: 0.8962450478690309.
#Trial 26 finished with value: 0.9415764295392566 and parameters: {'discount': 0.9, 'max_grad_norm': 0.6, 'gae_lambda': 0.92, 'lr': 0.0013261687855477475, 'entropy_coef': 4.181779221852914e-08, 'entropy_decay': 0.0011045551930278221, 'procs': 4, 'value_loss_coef': 0.1723433395418726, 'actor_hidden_size': 64, 'critic_hidden_size': 128, 'feature_hidden_size': 64, 'clip_eps': 0.2035398350991972, 'batch_size': 256, 'epochs': 5, 'dissim_coef': 0.025877235929188423, 'ssp_dim': 55, 'ssp_h': 0.18790019414748999}. Best is trial 26 with value: 0.9415764295392566.
env = 'maze-sample-5x5-v0'
n_frames = 20000
algos = np.array(['ppo'])
wrappers = np.array(['ssp'])#, 'ssp'])
n_seeds = 2
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
            initial_params={
                "gae_lambda": 0.8,
                "procs": 4,
                "entropy_coef": entropy_coef,
                "entrpoy_decay": entrpoy_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "feature_size": 128,
                "dissim_coef": 0.0,
                "actor_hidden_size": 64,
                "clip_eps": 0.2,
                "critic_hidden_size": 64,
                }
            
        if 'ssp' in wrap:
            inwrap = 'ssp-auto'
            discount= 0.99
            max_grad_norm=0.6
            gae_lambda=0.92
            lr =0.0013261687855477475
            entropy_coef=4.181779221852914e-08
            entropy_decay=0.0011045551930278221
            procs=4
            value_loss_coef= 0.1723433395418726
            actor_hidden_size= 64
            critic_hidden_size= 128
            feature_hidden_size= 64
            clip_eps=0.2035398350991972
            batch_size=256
            epochs= 5
            
            ssp_dim= 55
            ssp_h= 0.18790019414748999
            disism_coef = 0.0
            
            initial_params={
                "gae_lambda": gae_lambda,
                "procs": procs,
                "entropy_coef": entropy_coef,
                "entrpoy_decay": entropy_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "ssp_dim": ssp_dim,
                "ssp_h": ssp_h,
                "feature_size": feature_hidden_size,#"dissim_coef": disism_coef,
                "actor_hidden_size": actor_hidden_size,
                "clip_eps": clip_eps,
                "critic_hidden_size": critic_hidden_size,
                }
            
        study = optim(env,algo,inwrap,'flat',n_frames, n_seeds,n_trials, 
                            domain_dim=2,initial_params=initial_params )
        params[algo + '-' + wrap] = study.best_params
        print(study.best_params)
 
        
        
            

                
            
        
            

