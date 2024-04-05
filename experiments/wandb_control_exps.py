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
entity, project = "nicole-s-dumont", "ssp-rl"
envs = ["CartPole-v1", "MountainCar-v0","MountainCarContinuous-v0", "Pendulum-v1","Acrobot-v1","LunarLander-v2","LunarLanderContinuous-v2"]
obs_types = ["default-obs","ssp-obs","new-ssp-obs","tuned-ssp-obs"]
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
for r in runs:
    # if any([e in r.tags for e in envs]) and any([o in r.tags for o in obs_types]):
    for i,env in enumerate(envs):
        for obs_type in obs_types:
            try:
                args = r.metadata['args']
            except:
                pass
            argsdict ={k: v for k, v in zip(args[0::2], args[1::2])} 
            if (argsdict['--env'] == env) and (obs_type in r.tags):
                if ("Continuous" not in env) or ( ("Continuous" in env) and (argsdict['--algo'] == "sac" )):
                    try:
                        summary_list.append([env, obs_type, r.summary['global_step'], r.summary['rollout/ep_rew_mean']])#, r.summary['eval/mean_reward']])
                        rs = np.array(r.history()['rollout/ep_rew_mean'])
                        if (i==0) or (np.nanmax(rs) > max_rews[i]):
                            max_rews[i] = np.nanmax(rs)
                        if (i==0) or (np.nanmin(rs) < min_rews[i]):
                            min_rews[i] = np.nanmin(rs)
                    except:
                        pass
                
                
       
runs_summary = pd.DataFrame(summary_list, columns=["env","type","global_step","rollout/ep_rew_mean"])#,"eval/mean_reward"])
df2 = runs_summary.groupby(['env','type'], as_index=False).agg({'rollout/ep_rew_mean': ['mean', 'sem']})

resdict = {}

resdict['CartPole-default'] = np.array(runs_summary[(runs_summary['env']=="CartPole-v1") & (runs_summary['type']=="default-obs")]["rollout/ep_rew_mean"])[:,None]
resdict['CartPole-ssp'] = np.array(runs_summary[(runs_summary['env']=="CartPole-v1") & (runs_summary['type']=="new-ssp-obs")]["rollout/ep_rew_mean"])[:,None]

resdict['MountainCar-default'] = np.array(runs_summary[(runs_summary['env']=="MountainCar-v0") & (runs_summary['type']=="default-obs")]["rollout/ep_rew_mean"])[:,None]
resdict['MountainCar-ssp'] = np.array(runs_summary[(runs_summary['env']=="MountainCar-v0") & (runs_summary['type']=="new-ssp-obs")]["rollout/ep_rew_mean"])[:,None]

resdict['MountainCarContinuous-default'] = np.array(runs_summary[(runs_summary['env']=="MountainCarContinuous-v0") & (runs_summary['type']=="default-obs")]["rollout/ep_rew_mean"])[:,None]
resdict['MountainCarContinuous-ssp'] = np.array(runs_summary[(runs_summary['env']=="MountainCarContinuous-v0") & (runs_summary['type']=="ssp-obs")]["rollout/ep_rew_mean"])[:,None]

resdict['LunarLander-default'] = np.array(runs_summary[(runs_summary['env']=="LunarLander-v2") & (runs_summary['type']=="default-obs")]["rollout/ep_rew_mean"])[:,None]
resdict['LunarLander-ssp'] = np.array(runs_summary[(runs_summary['env']=="LunarLander-v2") & (runs_summary['type']=="ssp-obs")]["rollout/ep_rew_mean"])[:,None]

resdict['LunarLanderContinuous-default'] = np.array(runs_summary[(runs_summary['env']=="LunarLanderContinuous-v2") & (runs_summary['type']=="default-obs")]["rollout/ep_rew_mean"])[:,None]
resdict['LunarLanderContinuous-ssp'] = np.array(runs_summary[(runs_summary['env']=="LunarLanderContinuous-v2") & (runs_summary['type']=="ssp-obs")]["rollout/ep_rew_mean"])[:,None]

resdict['Pendulum-default'] = np.array(runs_summary[(runs_summary['env']=="Pendulum-v1") & (runs_summary['type']=="default-obs")]["rollout/ep_rew_mean"])[:,None]
resdict['Pendulum-ssp'] = np.array(runs_summary[(runs_summary['env']=="Pendulum-v1") & (runs_summary['type']=="ssp-obs")]["rollout/ep_rew_mean"])[:,None]

resdict['Acrobot-default'] = np.array(runs_summary[(runs_summary['env']=="Acrobot-v1") & (runs_summary['type']=="default-obs")]["rollout/ep_rew_mean"])[:,None]
resdict['Acrobot-ssp'] = np.array(runs_summary[(runs_summary['env']=="Acrobot-v1") & (runs_summary['type']=="ssp-obs")]["rollout/ep_rew_mean"])[:,None]

aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
  resdict, metrics.aggregate_iqm, reps=50000)
# df2 = runs_summary.groupby(['env','type'], as_index=False).agg({'rollout/ep_rew_mean': ['median', 'std']})
#     columns=["type","CartPole-v1", "MountainCar-v0", "Pendulum-v1","Acrobot-v1","LunarLander-v2"])



improvprobs = {}
for e in ['CartPole', 'MountainCar', 'MountainCarContinuous', 'LunarLander', 'LunarLanderContinuous','Pendulum','Acrobot']:
    improvprobs[e] = metrics.probability_of_improvement(resdict[e+'-ssp'], resdict[e+'-default'])

from scipy.stats import mannwhitneyu
improvprobs = {}
pvals = {}
for e in ['CartPole', 'MountainCar', 'MountainCarContinuous', 'LunarLander', 'LunarLanderContinuous','Pendulum','Acrobot']:
    s,p = mannwhitneyu(resdict[e+'-ssp'], resdict[e+'-default'], alternative='greater')
    improvprobs[e] = s/ (len(resdict[e+'-ssp']) * len(resdict[e+'-default']))
    pvals[e] = p
    
cohen_ds = {}
cohen_ds_size = {}
for e in [ 'MountainCar', 'MountainCarContinuous', 'LunarLander', 'LunarLanderContinuous','Pendulum','Acrobot']:
    c0 = resdict[e+'-ssp']
    c1 = resdict[e+'-default']
    cohen_ds[e]= (np.mean(c0) - np.mean(c1)) / (np.sqrt((np.std(c0, ddof=1)**2 + np.std(c1, ddof=1)**2) / 2.0))
    if cohen_ds[e] < 0.2:
        cohen_ds_size[e] = 'null'
    elif cohen_ds[e] < 0.5:
        cohen_ds_size[e] = 'small'
    elif cohen_ds[e] < 0.8:
        cohen_ds_size[e] = 'medium'
    else:
        cohen_ds_size[e] = 'large'
    
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