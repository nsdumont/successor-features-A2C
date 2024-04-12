import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from hyperparam_opt import optim
import utils
import numpy as np

#image_cm
#Trial 63 finished with value: 0.7247721403837204 and parameters: {'max_grad_norm': 1, 'gae_lambda': 0.95, 'lr': 1.551659964600454e-05, 'entropy_coef': 0.05003474519593319, 'entropy_decay': 3.124098155754289e-08, 'procs': 4, 'value_loss_coef': 0.34803454412571305, 'actor_hidden_size': 64, 'critic_hidden_size': 32, 'feature_hidden_size': 32}. Best is trial 63 with value: 0.7247721403837204.            
            
                    
#ssp-xy_latent
#Trial 4 finished with value: 0.9458333253860474 and parameters: {'max_grad_norm': 0.3, 'gae_lambda': 1.0, 'lr': 0.0017270083130236365, 'entropy_coef': 2.4679009323089367e-08, 'entropy_decay': 2.909523775296993e-05, 'procs': 1, 'value_loss_coef': 0.16112834586237457, 'actor_hidden_size': 32, 'critic_hidden_size': 64, 'feature_hidden_size': 32, 'ssp_dim': 129, 'ssp_h_0': 18.43163107671836, 'ssp_h_1': 17.0904786808492, 'ssp_h_2': 0.018008208823351766}. Best is trial 4 with value: 0.9458333253860474.
#ssp-view_latent
#Trial 8 finished with value: 0.9557291766007742 and parameters: {'max_grad_norm': 1, 'gae_lambda': 0.98, 'lr': 0.004817046936674357, 'entropy_coef': 1.1290096641176217e-07, 'entropy_decay': 3.0713000212417454e-06, 'procs': 6, 'value_loss_coef': 0.18182088220276893, 'actor_hidden_size': 256, 'critic_hidden_size': 32, 'feature_hidden_size': 256, 'ssp_dim': 33, 'ssp_h_0': 0.0036358926940071585, 'ssp_h_1': 0.1003837613424786, 'ssp_h_2': 0.18939089483802682}. Best is trial 8 with value: 0.9557291766007742.Trial 8 finished with value: 0.9557291766007742 
env = "MiniGrid-Empty-6x6-v0"
n_frames = 40000
algos = ['sr']
types = ['image_cm','image_icm','image_latent','image_aenc','image_lap',
                  'ssp-xy_latent','ssp-view_latent','ssp-learn_latent']#, 'ssp']
n_seeds = 3
n_trials=100
params = {}
for algo in algos:
    for exp in types:
        wrap =  exp.split("_")[0]
        feature_learn = exp.split("_")[1]
        if wrap=='image':
            inwrap='none'
            input_type='image'
        elif wrap=='ssp-learn':
            inwrap='xy'
            input_type='ssp'
        else:
            inwrap=wrap
            input_type='flat'
            
    
        study = optim(env,algo,inwrap,input_type,n_frames, n_seeds,n_trials, 
                            domain_dim=3, feature_learn=feature_learn )
        params[algo + '-' + wrap] = study.best_params
        
print(params)
 
        
        
            

# import optuna
# from optuna.visualization import plot_contour
# from optuna.visualization import plot_edf
# from optuna.visualization import plot_intermediate_values
# from optuna.visualization import plot_optimization_history
# from optuna.visualization import plot_parallel_coordinate
# from optuna.visualization import plot_param_importances
# from optuna.visualization import plot_rank
# from optuna.visualization import plot_slice
# from optuna.visualization import plot_timeline
# loaded_study = optuna.load_study(study_name="MiniGrid-Empty-6x6-v0_sr_ssp-view_24-04-11-17-55-05", storage="sqlite:///MiniGrid-Empty-6x6-v0_sr_ssp-view_24-04-11-17-55-05.db")
