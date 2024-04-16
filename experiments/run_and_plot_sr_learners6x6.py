import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils

models = ['plot_sr_image_cm','plot_sr_image_icm', 'plot_sr_image_aenc',
          'plot_sr_image_lap','plot_sr_image_latent',#'plot_sr_ssp-xy_none','plot_sr_ssp-view_none',
          'plot_sr_ssp-xy_latent','plot_sr_ssp-view_latent']
#'plot_sr_xy_icm'] # learn ssp encoding, no other feature transform
models = ['plot_sr_ssp-view_latent']
# python hyperparam_opt.py --env MiniGrid-FourRooms-v0 --algo ppo --wrapper ssp-view --input flat --n-seeds 3 --frames 1000000 --n-trials 100 --domain-dim 3 --other-args '{"wrapper_args": {"ignore": ["FLOOR"]}}' |& tee optuna_minigrid_fourrooms.txt

# python hyperparam_opt.py --env maze-sample-5x5-v0 --algo ppo --wrapper one-hot --input flat --n-seeds 3 --frames 2000 --domain-dim 3 
envs = [ "MiniGrid-Empty-6x6-v0"] 
# "MiniGrid-Empty-8x8-v0",
#                  "MiniGrid-Empty-16x16-v0", 
#                  "MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-6x6-v0"])
n_frames = [60000,80000,160000,100000,100000]
n_seeds=5
all_seeds = np.arange(n_seeds)#np.random.randint(0,10000,n_seeds)
replace_existing= True

ssp_h = np.array([1,1,0.1])[:,None]
#image_cm
#Trial 63 finished with value: 0.7247721403837204 and parameters: {'max_grad_norm': 1, 'gae_lambda': 0.95, 'lr': 1.551659964600454e-05, 'entropy_coef': 0.05003474519593319, 'entropy_decay': 3.124098155754289e-08, 'procs': 4, 'value_loss_coef': 0.34803454412571305, 'actor_hidden_size': 64, 'critic_hidden_size': 32, 'feature_hidden_size': 32}. Best is trial 63 with value: 0.7247721403837204.            
            
                    
#ssp-xy_latent
#Trial 4 finished with value: 0.9458333253860474 and parameters: {'max_grad_norm': 0.3, 'gae_lambda': 1.0, 'lr': 0.0017270083130236365, 'entropy_coef': 2.4679009323089367e-08, 'entropy_decay': 2.909523775296993e-05, 'procs': 1, 'value_loss_coef': 0.16112834586237457, 'actor_hidden_size': 32, 'critic_hidden_size': 64, 'feature_hidden_size': 32, 'ssp_dim': 129, 'ssp_h_0': 18.43163107671836, 'ssp_h_1': 17.0904786808492, 'ssp_h_2': 0.018008208823351766}. Best is trial 4 with value: 0.9458333253860474.
#ssp-view_latent
#Trial 8 finished with value: 0.9557291766007742 and parameters: {'max_grad_norm': 1, 'gae_lambda': 0.98, 'lr': 0.004817046936674357, 'entropy_coef': 1.1290096641176217e-07, 'entropy_decay': 3.0713000212417454e-06, 'procs': 6, 'value_loss_coef': 0.18182088220276893, 'actor_hidden_size': 256, 'critic_hidden_size': 32, 'feature_hidden_size': 256, 'ssp_dim': 33, 'ssp_h_0': 0.0036358926940071585, 'ssp_h_1': 0.1003837613424786, 'ssp_h_2': 0.18939089483802682}. Best is trial 8 with value: 0.9557291766007742.Trial 8 finished with value: 0.9557291766007742 
for j, env in enumerate(envs):
    for i,model_name in enumerate(models):
        print("Starting "+ model_name )
        algo = model_name.split("_")[1]
        wrapper = model_name.split("_")[2]
        feature_learn = model_name.split("_")[3]
        
        input_type = 'flat'
        feature_hidden_size=128
        critic_hidden_size=256
        entropy_coef=0.001
        entropy_decay=0.1
        normalize=True
        inwrapper=wrapper
        if feature_learn=='none':
            normalize=False
            critic_hidden_size=128
        if algo=='a2c':
            critic_hidden_size=64  
            entropy_coef=0.0005
            entropy_decay=0.0
            normalize=False
        if wrapper=='image':
            inwrapper="none"
            input_type='image'
        if wrapper=='xy':
            inwrapper="xy"
            input_type='ssp'
            
        if ('cm' in feature_learn):#
            max_grad_norm= 1
            gae_lambda= 0.95
            lr= 1.551659964600454e-05
            entropy_coef= 0.05003474519593319
            entropy_decay= 3.124098155754289e-08
            procs= 4
            value_loss_coef=0.34803454412571305
            actor_hidden_size= 64
            critic_hidden_size= 32
            feature_hidden_size= 32
        if (wrapper=='ssp-xy') and (feature_learn=='latent'):
            max_grad_norm= 0.3
            gae_lambda= 1.0
            lr= 0.0017270083130236365
            entropy_coef= 2.4679009323089367e-08
            entropy_decay= 2.909523775296993e-05
            procs= 4 #**
            value_loss_coef= 0.16112834586237457
            actor_hidden_size= 32
            critic_hidden_size= 64
            feature_hidden_size= 32
            ssp_dim= 129
            ssp_h = np.array([ 18.43163107671836, 17.0904786808492, 0.018008208823351766])[:,None]
        if (wrapper=='ssp-view') and (feature_learn=='latent'):
            max_grad_norm= 1
            gae_lambda= 0.98
            lr= 0.004817046936674357
            entropy_coef= 1.1290096641176217e-07
            entropy_decay= 3.0713000212417454e-06
            procs= 6
            value_loss_coef=0.18182088220276893
            actor_hidden_size= 256
            critic_hidden_size= 32
            feature_hidden_size= 256
            ssp_dim= 33
            ssp_h = np.array([ 1.2194623317407103, 0.12499819462090338,  0.019681173426129097])[:,None]
        
            ssp_h = np.array([ 1, 1,  0.01])[:,None]
        for seed in all_seeds:
            seed = int(seed)
            _model_name =  f"{env}_" + model_name + f"_{seed}" 
            
            model_dir = utils.get_model_dir(_model_name)
            if os.path.exists(model_dir):
                if replace_existing:
                    for f in os.listdir(model_dir):
                        os.remove(os.path.join(model_dir, f))
                    os.rmdir(model_dir)
                else:
                    pass
                
            run(algo = algo, wrapper=inwrapper, model = _model_name, 
                seed=seed,input=input_type, normalize_embeddings=normalize,
                  env=env, frames=n_frames[j],feature_learn=feature_learn,
                  max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,lr=lr,
                  entropy_coef=entropy_coef, entropy_decay=entropy_decay,
                  procs=procs,value_loss_coef=value_loss_coef,
                  actor_hidden_size=actor_hidden_size,critic_hidden_size=critic_hidden_size,feature_hidden_size=feature_hidden_size,
                  ssp_dim=ssp_dim,ssp_h=ssp_h,
                  verbose=False,recurrence=1)
            print("Finished "+ _model_name )
     
models = ['plot_sr_image_cm','plot_sr_image_icm', 'plot_sr_image_aenc',
          'plot_sr_image_lap','plot_sr_image_latent',#'plot_sr_ssp-xy_none','plot_sr_ssp-view_none',
          'plot_sr_ssp-xy_latent','plot_sr_ssp-view_latent']
linestys = {'image': '-', 'ssp-xy':'--', 'ssp-view':':', 'xy':'--'}
cols= {'cm': utils.reds[0], 'icm': utils.oranges[0],'lap': utils.blues[0],
       'aenc':utils.greens[0], 'latent': utils.yellows[0],
       'none': utils.purples[0],}

for j, env in enumerate(envs):
    fig=plt.figure(figsize=(8,4))
    for i,model_name in enumerate(models):
        df = pd.DataFrame()
    
        for seed in range(0,n_seeds):
            model_dir = utils.get_model_dir( f"{env}_" + model_name +  '_' + str(seed))
            data = pd.read_csv(model_dir + "/log.csv")
            #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
            data['return_mean'] = pd.to_numeric(data['return_mean'], errors='coerce')
            data['frames'] = pd.to_numeric(data['frames']).astype(float)
            data['return_rollingavg'] = data['return_mean'].rolling(50).mean()
            df = df._append(data)
           
            
        
        #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
        input_type = model_name.split("_")[2]
        learner = model_name.split("_")[3]
        
        if model_name.split("_")[1]=='a2c':
            sns.lineplot(x="frames", y='return_mean', label=", ".join(model_name.split("_")[1:]),errorbar=None,#'ci',
                         data=df, alpha=0.8, linestyle=linestys[input_type], color='k')
        else:
            sns.lineplot(x="frames", y='return_mean', label=", ".join(model_name.split("_")[1:]),errorbar=None,#'ci',
                         data=df, alpha=0.8, linestyle=linestys[input_type], color=cols[learner])
    plt.legend(frameon=True,edgecolor='white')
    plt.xlabel("Frames observed")
    plt.ylabel("Average Return")
    plt.title(env)
    
    
for j, env in enumerate(envs):
    fig=plt.figure(figsize=(8,4))
    for i,model_name in enumerate(models):
    
        seed =1
        model_dir = utils.get_model_dir( f"{env}_" + model_name +  '_' + str(seed))
        data = pd.read_csv(model_dir + "/log.csv")
        #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
        data['return_mean'] = pd.to_numeric(data['return_mean'], errors='coerce')
        data['frames'] = pd.to_numeric(data['frames']).astype(float)
        data['return_rollingavg'] = data['return_mean'].rolling(50).mean()
                
        
        #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
        input_type = model_name.split("_")[2]
        learner = model_name.split("_")[3]
        
        if model_name.split("_")[1]=='a2c':
            sns.lineplot(x="frames", y='return_rollingavg', label=", ".join(model_name.split("_")[1:]),errorbar=None,#'ci',
                         data=data, alpha=0.8, linestyle=linestys[input_type], color='k')
        else:
            sns.lineplot(x="frames", y='return_rollingavg', label=", ".join(model_name.split("_")[1:]),errorbar=None,#'ci',
                         data=data, alpha=0.8, linestyle=linestys[input_type], color=cols[learner])
    plt.legend(frameon=True,edgecolor='white')
    plt.xlabel("Frames observed")
    plt.ylabel("Average Return")
    plt.title(env)
# utils.save(fig,"figures/sr_" + env_name + ".pdf")
# utils.save(fig,"figures/sr_" + env_name + "_ma50.png")


