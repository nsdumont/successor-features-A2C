import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils
import numpy as np
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-DoorKey-8x8-v0 --frames 100000 --wrapper ssp-view --input flat --plot True --procs 1 --frames-per-proc 100 ', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)

envs = np.array(['maze-sample-5x5-v0',  'maze-sample-6x6-v0', 'maze-sample-7x7-v0',
        'maze-sample-8x8-v0', 'maze-sample-9x9-v0', 'maze-sample-10x10-v0',
        ])
envs = np.array(['maze-sample-5x5-v0'])
n_frames = 20000
# envs = ['MiniGrid-Empty-8x8-v0', 'MiniGrid-Empty-16x16-v0', 'MiniGrid-FourRooms-v0',
#         'MiniWorld-TMazeLeft-v0', 'MiniWorld-YMazeLeft-v0', 
#         'MiniWorld-MazeS2-v0', 'MiniWorld-MazeS3-v0', 'MiniWorld-MazeS3Fast-v0']#,
      #  'maze-sample-5x5','maze-sample-10x10-v0','maze-sample-100x100-v0']
algos = np.array(['ppo'])
wrappers = np.array(['xy', 'ssp'])
n_seeds = 5
all_models = []
clip_eps=0.2
for i, env in enumerate(envs):
    models = []
    for algo in algos:

        for wrap in wrappers:
            models.append(env + '_' + algo + '_' + wrap )
            for seed in range(n_seeds):
                model_name = env + '_' + algo + '_' + wrap + '_' + str(seed)
                
                model_dir = utils.get_model_dir(model_name)
                if os.path.exists(model_dir):
                    for f in os.listdir(model_dir):
                        os.remove(os.path.join(model_dir, f))
                    os.rmdir(model_dir)
                    
                if wrap=='xy':
                    inwrap='none'
                    dissim_coef= 0
                    lr=0.0001
                    entropy_coef=0.002
                    entrpoy_decay=0.01
                    batch_size=256
                    epochs=4
                if 'ssp' in wrap:
                    dissim_coef =0.0#float(wrap.split('_')[1])
                    inwrap='ssp-auto'
                    lr=0.001
                    entropy_coef=0.0002
                    entrpoy_decay=0.01
                    batch_size=64
                    epochs=1
                

                    
                run( env=env, seed=seed, algo = algo, wrapper=inwrap, model = model_name, frames=n_frames,
                    lr=lr, gae_lambda=0.8, procs=5, entropy_coef=entropy_coef, 
                    entrpoy_decay=entrpoy_decay, batch_size=batch_size, clip_eps=clip_eps, epochs=epochs,
                    optim_eps=1e-9, n_test_episodes=0, ssp_h=1,feature_size=128,dissim_coef=dissim_coef,
                    input='flat',
                      verbose=False);
            
    all_models.append(models)
            


fig, axs =plt.subplots(len(envs),1,figsize=(7,2*len(envs)))
axs = [axs]
linestys = {'ppo': '--', 'a2c':'-'}
cols= { 'xy': utils.oranges[0],'ssp': utils.blues[0]}
for j,env in enumerate(envs):
    for i,model_name in enumerate(all_models[j]):
        df = pd.DataFrame()
        for seed in range(1,n_seeds):
            model_dir = utils.get_model_dir(model_name +  '_' + str(seed))
            data = pd.read_csv(model_dir + "/log.csv")
            #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
            data = data[pd.to_numeric(data['return_mean'], errors='coerce').notnull()]
            data['return_mean'] = pd.to_numeric(data['return_mean'])
            data['frames'] = pd.to_numeric(data['frames'])
            df = df._append(data)
        
        model_name_split = model_name.split("_")
        input_type =model_name_split[2]
        algo = model_name_split[1]
        # if len(model_name_split)>3:
        #     label=algo.upper() + "-" + input_type + " (c=" + model_name_split[3] +")"
        # else:
        label=algo.upper() + "-" + input_type
        sns.lineplot(x="frames", y='return_mean', label=label,errorbar='ci',
                     data=df, alpha=0.8, ax=axs[j])#, linestyle=linestys[input_type])#, color=cols[algo])
    axs[j].set_title("2D Maze (" + env.split('-')[-2] + ")")
    axs[j].set_xlabel("Frames observed")
    axs[j].set_ylabel("Average Return")
axs[0].legend(loc='lower right', facecolor='white', framealpha=1, frameon = True, edgecolor='none')
# utils.save(fig,'figures/' + env + '.pdf')