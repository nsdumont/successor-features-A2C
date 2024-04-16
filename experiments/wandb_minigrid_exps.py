import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import sys,os
os.chdir("..")
import utils


api = wandb.Api()
entity, project = "nicole-s-dumont", "sb3"
envs = ["MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-16x16-v0", "MiniGrid-Unlock-v0" , "MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-6x6-v0"   ]
obs_types = ["default-obs","ssp-obs"]
summary_list = []
max_rews = np.zeros(len(envs))
min_rews = np.zeros(len(envs))
# for env in envs:
#     for obs_type in obs_types:
#         runs = api.runs(entity + "/" + project,
#                    filters={"state":"finished",
#                             "$and": [{"tags": {"$in": env}}, {"tags": {"$in": obs_type}}]})
#         for r in runs:
#             try:
#                 summary_list.append([env, obs_type, r.summary['global_step'], r.summary['rollout/ep_rew_mean']])#, r.summary['eval/mean_reward']])
#             except:
#                 pass
# runs_summary = pd.DataFrame(summary_list, columns=["env","type","global_step","rollout/ep_rew_mean"])#,"eval/mean_reward"])

# df2 = runs_summary.groupby(['env','type'], as_index=False).agg({'rollout/ep_rew_mean': ['mean', 'std']})
runs = api.runs(entity + "/" + project,
           filters={"state":"finished"})
all_results = {}
all_frames = {}
for r in runs:
    # if any([e in r.tags for e in envs]) and any([o in r.tags for o in obs_types]):
    for i,env in enumerate(envs):
        for obs_type in obs_types:
            if env not in all_results:
                all_results[env] = {}
            if obs_type not in all_results[env]:
                all_results[env][obs_type] = []
            try:
                args = r.metadata['args']
            except:
                pass
            argsdict ={k: v for k, v in zip(args[0::2], args[1::2])} 
            if (argsdict['--env'] == env) and (obs_type in r.tags):
                try:
                    summary_list.append([env, obs_type, r.summary['global_step'], r.summary['rollout/ep_rew_mean']])#, r.summary['eval/mean_reward']])
                    all_results[env][obs_type].append( np.array(r.history()['rollout/ep_rew_mean']) )
                    if env not in all_frames:
                        all_frames[env] = np.array(r.history()['global_step'])
                except:
                    pass
                
                
       
runs_summary = pd.DataFrame(summary_list, columns=["env","type","global_step","rollout/ep_rew_mean"])#,"eval/mean_reward"])
df2 = runs_summary.groupby(['env','type'], as_index=False).agg({'rollout/ep_rew_mean': ['mean', 'sem']})

resdict = {}
for i,env in enumerate(envs):
    for obs_type in obs_types:
            resdict[env + '-' + obs_type] =np.array(runs_summary[(runs_summary['env']==env) & (runs_summary['type']==obs_type)]["rollout/ep_rew_mean"])[:,None]
            all_results[env][obs_type] = np.array(all_results[env][obs_type].copy())
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
  resdict, metrics.aggregate_iqm, reps=50000)
# df2 = runs_summary.groupby(['env','type'], as_index=False).agg({'rollout/ep_rew_mean': ['median', 'std']})
#     columns=["type","CartPole-v1", "MountainCar-v0", "Pendulum-v1","Acrobot-v1","LunarLander-v2"])



improvprobs = {}
for e in envs:
    improvprobs[e] = metrics.probability_of_improvement(resdict[e+'-ssp-obs'], resdict[e+'-default-obs'])
    
from scipy.stats import mannwhitneyu
improvprobs = {}
pvals = {}
for e in envs:
    s,p = mannwhitneyu(resdict[e+'-ssp-obs'], resdict[e+'-default-obs'], alternative='greater')
    improvprobs[e] = s/ (len(resdict[e+'-ssp-obs']) * len(resdict[e+'-default-obs']))
    pvals[e] = p
    
cohen_ds = {}
cohen_ds_size = {}
for e in envs:
    c0 = resdict[e+'-ssp-obs']
    c1 = resdict[e+'-default-obs']
    cohen_ds[e]= (np.mean(c0) - np.mean(c1)) / (np.sqrt((np.std(c0, ddof=1)**2 + np.std(c1, ddof=1)**2) / 2.0))
    if cohen_ds[e] < 0.2:
        cohen_ds_size[e] = 'null'
    elif cohen_ds[e] < 0.5:
        cohen_ds_size[e] = 'small'
    elif cohen_ds[e] < 0.8:
        cohen_ds_size[e] = 'medium'
    else:
        cohen_ds_size[e] = 'large'
        
        
from rliable import library as rly
from rliable import metrics

fig, axs =plt.subplots(len(envs), 1, figsize=(6.5,1.6*len(envs)), sharey=True,constrained_layout=True)
cols= { 'default-obs': utils.oranges[0],'ssp-obs': utils.blues[0]}
for i,env in enumerate(envs):
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                               for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(
          all_results[env], iqm, reps=10000)
    utils.plot_sample_efficiency_curve(
        all_frames[env], iqm_scores, iqm_cis, algorithms=obs_types,
        xlabel=None,ylabel=None, ax=axs[i], colors=cols)
    
    # axs[j].hlines(optim_rewards[env],0,all_frames[-1], colors='gray',linestyles='dashed',label='Optimal')
    if i==0:
        axs[i].legend(loc='lower right', facecolor='white', framealpha=1, frameon = True, edgecolor='none')
    else:
        axs[i].legend([],[], frameon=False)

    # axs[j].set_title("2D Maze (" + env.split('-')[-2] + ")")
    axs[i].set_title(env)
    axs[i].ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    axs[i].set_ylabel("Episodic Reward",fontsize=11)
    if i==(len(envs)-1):
        axs[i].set_xlabel("Frames Observed",fontsize=11)


utils.save(fig,"figures/minigrid_iqm_return.pdf")
utils.save(fig,"figures/minigrid_iqm_return.png")
    
# runs = api.runs(entity + "/" + project) 

# summary_list, config_list, name_list = [], [], []
# for run in runs: 
#     if run.state == "finished":
#         # .summary contains the output keys/values for metrics like accuracy.
#         #  We call ._json_dict to omit large files 
#         summary_list.append(run.summary._json_dict)
    
#         # .config contains the hyperparameters.
#         #  We remove special values that start with _.
#         config_list.append(
#             {k: v for k,v in run.config.items()
#              if not k.startswith('_')})
    
#         # .name is the human-readable name of the run.
#         name_list.append(run.name)

# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
#     })



# run = api.runs(path="nicole-s-dumont/ssp-rl/CartPole-v1__ppo__4__1709150584")
               
# runs = api.runs(entity + "/" + project,
#                filters={"state":"finished", "tags": {"$in": ["CartPole-v1"]}} )

# for r in runs:
# if run.state == "finished":
#     for i, row in run.history().iterrows():
#         print(row["_timestamp"], row["accuracy"])