import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
import utils


api = wandb.Api()
entity, project = "nicole-s-dumont", "ssp-rl"
envs = ["CartPole-v1", "MountainCar-v0", "Pendulum-v1","Acrobot-v1","LunarLander-v2"]
obs_types = ["default-obs","ssp-obs"]
summary_list = []
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
    for env in envs:
        for obs_type in obs_types:
            if (env in r.tags) and (obs_type in r.tags):
                try:
                    summary_list.append([env, obs_type, r.summary['global_step'], r.summary['rollout/ep_rew_mean']])#, r.summary['eval/mean_reward']])
                except:
                    pass
                
       
runs_summary = pd.DataFrame(summary_list, columns=["env","type","global_step","rollout/ep_rew_mean"])#,"eval/mean_reward"])

df2 = runs_summary.groupby(['env','type'], as_index=False).agg({'rollout/ep_rew_mean': ['mean', 'std']})
#     columns=["type","CartPole-v1", "MountainCar-v0", "Pendulum-v1","Acrobot-v1","LunarLander-v2"])

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