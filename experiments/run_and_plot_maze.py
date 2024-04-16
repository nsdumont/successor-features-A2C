import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils
import numpy as np
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-DoorKey-8x8-v0 --frames 100000 --wrapper ssp-view --input flat --plot True --procs 1 --frames-per-proc 100 ', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#
envs = np.array([ 'maze-sample-5x5-v0', 'maze-sample-6x6-v0', 'maze-sample-7x7-v0',
        'maze-sample-8x8-v0', 'maze-sample-9x9-v0', 'maze-sample-10x10-v0',
        'maze-sample-11x11-v0', 'maze-sample-12x12-v0','maze-sample-15x15-v0',
        'maze-sample-20x20-v0' ])
# optim_num_steps= dict(zip(envs,[15, 19, 31, 33, 19, 47, 61, 39, 79, 77]))
optim_rewards=dict(zip(envs,[0.9874, 0.9838,0.973, 0.9712, 0.9838, 0.9586, 0.9553719008264463, 0.97625, 0.9688, 0.9829]))
algos = np.array(['ppo'])
wrappers = np.array(['xy', 'one-hot','ssp', 'ssp-learn'])#

# envs = np.array([ 'maze-sample-5x5-v0'])
# wrappers = np.array(['ssp-learn'])#

discount = 0.99
n_seeds = 3
dissim_coef=0.0
replace_existing = True
for i, env in enumerate(envs):
    maze_size = int(env.split('-')[2].split('x')[0])
    if maze_size < 9:
        n_frames = 20000
    elif maze_size < 15:
        n_frames = 80000
    elif maze_size < 20:
        n_frames = 100000
    else:
        n_frames = 100000
    
    for algo in algos:
        for wrap in wrappers:
            for seed in range(n_seeds):
                model_name = env + '_' + algo + '_' + wrap + '_' + str(seed)
                
                model_dir = utils.get_model_dir(model_name)
                if os.path.exists(model_dir):
                    if replace_existing:
                        for f in os.listdir(model_dir):
                            os.remove(os.path.join(model_dir, f))
                        os.rmdir(model_dir)
                    else:
                        pass
                
                if (wrap=='xy') or (wrap=='one-hot'):
                    if wrap=='one-hot':
                        inwrap='one-hot'
                        #Trial 83 finished with value: 0.982803565405664 and parameters: 
                        max_grad_norm= 5
                        gae_lambda= 0.99
                        lr= 0.0011172563477130418
                        entropy_coef= 0.0034417738877241867
                        entropy_decay= 0.07359940720031101
                        procs= 4
                        value_loss_coef= 0.5089918537606997
                        actor_hidden_size= 32
                        critic_hidden_size= 256
                        feature_hidden_size= 32
                        clip_eps= 0.13178468542859872
                        batch_size= 64
                        epochs= 20
                    else:
                        inwrap='none'
                        input_type='flat'
                        dissim_coef= 0
                        # lr=0.0001
                        # entropy_coef=0.002
                        # entrpoy_decay=0.01
                        # batch_size=256
                        # epochs=4
                        
                        # With -0.1 every step
                        # discount=0.99
                        # max_grad_norm= 0.8
                        # gae_lambda= 0.95
                        # lr= 0.0010650071035089798
                        # entropy_coef= 0.000678150163515309
                        # entropy_decay= 9.347513374190233e-07
                        # procs= 6
                        # value_loss_coef= 0.14648442322655858
                        # actor_hidden_size= 128
                        # critic_hidden_size= 64
                        # feature_hidden_size= 128
                        # clip_eps= 0.17566918588142932
                        # batch_size= 32
                        # epochs= 5
                        
                        # with only final reward and fewer max steps
                        max_grad_norm= 5
                        gae_lambda= 0.95
                        lr= 0.0030972523257061122
                        entropy_coef= 2.136031953414213e-05
                        entropy_decay= 0.012867127279332572
                        procs=4
                        value_loss_coef= 0.672960306771515
                        actor_hidden_size= 64
                        critic_hidden_size= 64
                        feature_hidden_size= 64
                        clip_eps= 0.20845387344130875
                        batch_size= 64
                        epochs= 5
                    
                    ssp_dim=1
                    ssp_h=1
                if wrap=='ssp-learn':
                    inwrap = 'none'
                    input_type = 'ssp'
                    
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
                    ssp_h=1.0
                    
                    # max_grad_norm= 2
                    # gae_lambda= 0.95
                    # lr=2.2183220259410734e-05
                    # entropy_coef= 1.236685364126784e-05
                    # entropy_decay= 0.0015574029070002872
                    # procs= 4
                    # value_loss_coef= 0.1481032251855734
                    # actor_hidden_size= 256
                    # critic_hidden_size= 128
                    # feature_hidden_size= 128
                    # clip_eps= 0.014141657615793196
                    # batch_size= 16
                    # epochs= 5
                    # ssp_dim= 145
                    # ssp_h = 1.0
                    
                    # max_grad_norm= 2
                    # gae_lambda= 1.0
                    # lr=0.001939282869
                    # entropy_coef= 0.0002321296
                    # entropy_decay=0.00043517
                    # procs= 4
                    # value_loss_coef= 0.3766876
                    # actor_hidden_size= 32
                    # critic_hidden_size= 64
                    # feature_hidden_size= 32
                    # clip_eps= 0.51553
                    # batch_size= 512
                    # epochs= 20
                    # ssp_dim= 197
                    # ssp_h = 1.0
                elif 'ssp' in wrap:
                    # dissim_coef =0.0#float(wrap.split('_')[1])
                    inwrap='ssp-auto'
                    input_type = 'flat'
                    
                    # Found by trial and error
                    # discount=0.99
                    # max_grad_norm=10
                    # gae_lambda=0.95
                    # procs=5
                    # value_loss_coef=0.5
                    # actor_hidden_size= 64
                    # critic_hidden_size= 64
                    # feature_hidden_size= 128
                    # clip_eps=0.2
                    # ssp_dim = 151
                    # ssp_h = 1.0
                    # lr=0.001
                    # entropy_coef=0.0002
                    # entrpoy_decay=0.01
                    # batch_size=64
                    # epochs=1
                    # dissim_coef=0.0
                    
                    # with -0.1 each step
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
                    ssp_h=1# 0.18790019414748999*(maze_size/5)**2 # hurts performance?
                    # if wrap=='ssp0':
                    #     dissim_coef=0
                    # else:
                    #     dissim_coef= 0.025877235929188423
                    
                    #new: fewer max steps
                    # Trial 27 finished with value: 0.6600416832500035 and parameters: {
                    # max_grad_norm= 0.8
                    # gae_lambda= 0.9
                    # lr= 0.0003299826583861107
                    # entropy_coef= 0.0010538779659947858
                    # entropy_decay= 1.1752648239995013e-05
                    # procs= 1
                    # value_loss_coef= 0.21295888898313275
                    # actor_hidden_size= 256
                    # critic_hidden_size= 32
                    # feature_hidden_size= 128
                    # clip_eps= 0.34999590703584876
                    # batch_size= 32
                    # epochs= 20
                    # ssp_dim= 151
                    # ssp_h= 0.00033943615490787644
                    # dissim_coef=0

                    
                run( env=env, seed=seed, algo = algo, wrapper=inwrap, model = model_name, frames=n_frames,
                    n_test_episodes=0, input=input_type,verbose=False,
                      discount=discount, max_grad_norm=max_grad_norm, gae_lambda=gae_lambda,
                      lr=lr, entropy_coef=entropy_coef, entropy_decay=entropy_decay, procs=procs,
                      value_loss_coef=value_loss_coef, actor_hidden_size=actor_hidden_size, 
                      critic_hidden_size=critic_hidden_size, feature_hidden_size=feature_hidden_size,
                      clip_eps=clip_eps, batch_size=batch_size, epochs=epochs,
                      dissim_coef=dissim_coef,ssp_dim=ssp_dim, ssp_h=ssp_h);
            #optim_eps=1e-9, 
   
#
# envs = np.array(['maze-sample-5x5-v0', 'maze-sample-6x6-v0', 'maze-sample-7x7-v0',
#         'maze-sample-8x8-v0', 'maze-sample-9x9-v0', 'maze-sample-10x10-v0',
#          'maze-sample-12x12-v0','maze-sample-15x15-v0',
#         'maze-sample-20x20-v0' ])
wrappers = np.array(['one-hot','ssp-learn'])#

all_models = [[e + '_' + algos[0] + '_' + w for w in wrappers ] for e in envs]

if len(envs)==1:
    fig, axs =plt.subplots(1,1,figsize=(6.5,2))
    axs = [axs]
else:
    if True:
        fig, axs =plt.subplots(len(envs)//3, 3,figsize=(6.5,1.6*(len(envs)//3)), sharey=True,constrained_layout=True)
        axs = axs.reshape(-1)
    else:
        fig, axs =plt.subplots(len(envs), 1, figsize=(6.5,1.6*len(envs)), sharey=True,constrained_layout=True)

# linestys = {'ppo': '--', 'a2c':'-'}
cols= { 'xy': utils.oranges[0],'ssp': utils.blues[0]}
for j,env in enumerate(envs):
    for i,model_name in enumerate(all_models[j]):
        df = pd.DataFrame()

        for seed in range(0,n_seeds):
            try:
                model_dir = utils.get_model_dir(model_name +  '_' + str(seed))
                data = pd.read_csv(model_dir + "/log.csv")
                #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
                data = data[pd.to_numeric(data['return_mean'], errors='coerce').notnull()]
                data['return_mean'] = pd.to_numeric(data['return_mean'])
                data['frames'] = pd.to_numeric(data['frames']).astype(float)
                df = df._append(data)
            except:
                pass
        
        model_name_split = model_name.split("_")
        input_type =model_name_split[2]
        algo = model_name_split[1]
        # if len(model_name_split)>3:
        #     label=algo.upper() + "-" + input_type + " (c=" + model_name_split[3] +")"
        # else:
        label=input_type + ' obs'#algo.upper() + "-" + input_type
        sns.lineplot(x="frames", y='return_mean', label=label,errorbar='ci',
                     data=df, alpha=0.8, ax=axs[j])#, linestyle=linestys[input_type])#, color=cols[algo])
    axs[j].hlines(optim_rewards[env],0,df['frames'].iloc[-1], colors='gray',linestyles='dashed',label='Optimal')
    if (j==2) or (len(envs)==1):
        axs[j].legend(loc='lower right', facecolor='white', framealpha=1, frameon = True, edgecolor='none')
    else:
        axs[j].legend([],[], frameon=False)

    # axs[j].set_title("2D Maze (" + env.split('-')[-2] + ")")
    axs[j].set_title("Maze " + env.split('-')[-2].split('x')[0] + '$\\times$' + env.split('-')[-2].split('x')[1] )
    axs[j].ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    if (j%3 == 0) or (len(envs)==1):
        axs[j].set_ylabel("Episodic Reward",fontsize=11)
    if (j>=(len(envs)-3)) or (len(envs)==1):
        axs[j].set_xlabel("Frames Observed",fontsize=11)

# utils.save(fig,"figures/maze_mean_return.pdf")
# utils.save(fig,"figures/maze_mean_return.png")


# from rliable import library as rly
# from rliable import metrics
# from rliable import plot_utils
# if len(envs)==1:
#     fig, axs =plt.subplots(1,1,figsize=(6.5,2))
#     axs = [axs]
# else:
#     if True:
#         fig, axs =plt.subplots(len(envs)//3, 3,figsize=(6.5,1.6*(len(envs)//3)), sharey=True,constrained_layout=True)
#         axs = axs.reshape(-1)      
#     else:
#         fig, axs =plt.subplots(len(envs), 1, figsize=(6.5,1.6*len(envs)), sharey=True,constrained_layout=True)
# cols= { 'xy': utils.oranges[0],'ssp': utils.blues[0], 'ssp-learn': utils.greens[0]}
# final_results = {}
# for j,env in enumerate(envs):
#     results = {}
#     for i,model_name in enumerate(all_models[j]):
#         model_name_split = model_name.split("_")
#         input_type =model_name_split[2]
#         algo = model_name_split[1]
#         maze_size = int(env.split('-')[2].split('x')[0])
#         if maze_size < 9:
#             n_frames = 20000
#         elif maze_size < 15:
#             n_frames = 80000
#         elif maze_size < 20:
#             n_frames = 100000
#         else:
#             n_frames = 200000
        
#         for seed in range(0,n_seeds):
#             model_dir = utils.get_model_dir(model_name +  '_' + str(seed))
#             data = pd.read_csv(model_dir + "/log.csv")
#             #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
#             data = data[pd.to_numeric(data['return_mean'], errors='coerce').notnull()]
#             if seed==0:
#                 results[input_type] = np.zeros((n_seeds, 1, len(data['frames'])))
#                 if j==0:
#                     final_results[input_type] = np.zeros((n_seeds, len(envs)))
                    
#             results[input_type][seed,0,:] = pd.to_numeric(data['return_mean'])/optim_rewards[env]
#             final_results[input_type][seed,j]=results[input_type][seed,0,-1]
#             all_frames = pd.to_numeric(data['frames']).astype(float)

#     iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
#                                for frame in range(scores.shape[-1])])
#     iqm_scores, iqm_cis = rly.get_interval_estimates(
#           results, iqm, reps=10000)
#     utils.plot_sample_efficiency_curve(
#         all_frames, iqm_scores, iqm_cis, algorithms=wrappers,
#         xlabel=None,ylabel=None, ax=axs[j],  colors=cols)
    
#     # axs[j].hlines(optim_rewards[env],0,all_frames[-1], colors='gray',linestyles='dashed',label='Optimal')
#     if (j==2) or (len(envs)==1):
#         axs[j].legend(loc='lower right', facecolor='white', framealpha=1, frameon = True, edgecolor='none')
#     else:
#         axs[j].legend([],[], frameon=False)

#     # axs[j].set_title("2D Maze (" + env.split('-')[-2] + ")")
#     axs[j].set_title("Maze " + env.split('-')[-2].split('x')[0] + '$\\times$' + env.split('-')[-2].split('x')[1] )
#     axs[j].ticklabel_format(style='sci',scilimits=(0,0),axis='x')
#     if (j%3 == 0) or (len(envs)==1):
#         axs[j].set_ylabel("Episodic Reward",fontsize=11)
#     if (j>=(len(envs)-3)) or (len(envs)==1):
#         axs[j].set_xlabel("Frames Observed",fontsize=11)


# utils.save(fig,"figures/maze_iqm_return.pdf")
# utils.save(fig,"figures/maze_iqm_return.png")


# thresholds = np.linspace(0.0, 1.0, 100)
# score_distributions, score_distributions_cis = rly.create_performance_profile(
#     final_results, thresholds)
# fig, ax = plt.subplots(1,1,figsize=(6.5,2))
# plot_utils.plot_performance_profiles(
#   score_distributions, thresholds,
#   performance_profile_cis=score_distributions_cis,
#   colors=cols,
#   xlabel=r'Normalized Episodic Reward $(\tau)$', ax=ax)
# plt.legend(loc='lower right', facecolor='white', framealpha=1, frameon = True, edgecolor='none')
# fig.tight_layout()
# utils.save(fig,"figures/maze_performance_profiles.pdf")
# utils.save(fig,"figures/maze_performance_profiles.png")
# from scipy.stats import mannwhitneyu
# from rliable import library as rly
# from rliable import metrics

# resdict = {}
# for j,env in enumerate(envs):
#     for i,model_name in enumerate(all_models[j]):
#         final_rews = []
#         for seed in range(0,n_seeds):
#             try:
#                 model_dir = utils.get_model_dir(model_name +  '_' + str(seed))
#                 data = pd.read_csv(model_dir + "/log.csv")
#                 #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
#                 data = data[pd.to_numeric(data['return_mean'], errors='coerce').notnull()]
#                 final_rews.append(data['return_mean'].iloc[-1])
#             except:
#                 pass
#         resdict[model_name.split("_")[0] + '-' + model_name.split("_")[2]] = np.array(final_rews.copy())[:,None]
        
# aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
#   resdict, metrics.aggregate_iqm, reps=50000)

# improvprobs = {}
# pvals = {}
# for e in envs:
#     s,p = mannwhitneyu(resdict[e+'-ssp'], resdict[e+'-xy'], alternative='greater')
#     improvprobs[e] = s/ (len(resdict[e+'-ssp']) * len(resdict[e+'-xy']))
#     pvals[e] = p
    
# cohen_ds = {}
# cohen_ds_size = {}
# for e in envs:
#     c0 = resdict[e+'-ssp']
#     c1 = resdict[e+'-xy']
#     cohen_ds[e]= (np.mean(c0) - np.mean(c1)) / (np.sqrt((np.std(c0, ddof=1)**2 + np.std(c1, ddof=1)**2) / 2.0))
#     if cohen_ds[e] < 0.2:
#         cohen_ds_size[e] = 'null'
#     elif cohen_ds[e] < 0.5:
#         cohen_ds_size[e] = 'small'
#     elif cohen_ds[e] < 0.8:
#         cohen_ds_size[e] = 'medium'
#     else:
#         cohen_ds_size[e] = 'large'


