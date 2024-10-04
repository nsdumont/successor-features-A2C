import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils

models = ['sr_ssp-learn_cm','sr_ssp-learn_icm','sr_ssp-learn_latent','sr_ssp-learn_lap',
    'sr_image_cm','sr_image_icm', 'sr_image_aenc',
          'sr_image_lap','sr_image_latent',
          'sr_ssp-xy_icm','sr_ssp-view_icm',
          'sr_ssp-xy_cm', 'sr_ssp-view_cm',
          'sr_ssp-xy_latent', 'sr_ssp-view_latent',
          'sr_ssp-xy_lap', 'sr_ssp-view_lap']



env_name = "MiniGrid-Empty-Random-8x8-v0"
n_seeds=5

 
replace_existing=False
all_models = []
# runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-Random-6x6-v0 --frames 50000  --wrapper ssp-view --plot True --feature-learn none --critic-hidden-size 128 --entropy-coef 0.001 --entropy-decay 0.9  --input flat --model rand6x6_sr_ssp-view_none', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
for i,model_name in enumerate(models):
    print("Starting "+ model_name )
    algo = model_name.split("_")[0]
    wrapper = model_name.split("_")[1]
    feature_learn = model_name.split("_")[2]
    
    feature_hidden_size=128
    critic_hidden_size=256
    entropy_coef=0.001
    entropy_decay=0.1
    normalize=True
    # if feature_learn=='none':
    #     normalize=False
    #     critic_hidden_size=128
    if algo=='a2c':
        critic_hidden_size=64  
        entropy_coef=0.0005
        entropy_decay=0.0
        normalize=False
    if wrapper=='image':
        wrapper="none"
        input = "image"
    else:
        input="flat"
    if ('cm' in feature_learn):#
        entropy_coef = 0.01
        entropy_decay= 0.9

    
    for seed in range(n_seeds):
        _model_name =  f"{env_name}_" + model_name + f"_{seed}" 
        all_models.append(_model_name)
        model_dir = utils.get_model_dir(_model_name)
        if replace_existing:
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    os.remove(os.path.join(model_dir, f))
                os.rmdir(model_dir)
        run(algo = algo, wrapper=wrapper, model = _model_name, seed=seed, input=input,
            normalize_embeddings=normalize,feature_learn=feature_learn,
              env=env_name, frames=150000, entropy_coef=entropy_coef, entropy_decay=entropy_decay,
              feature_hidden_size=feature_hidden_size,critic_hidden_size=critic_hidden_size,verbose=False)
    
    print("Finished "+ model_name )




# all_models = [f"{env_name}_{m}_{seed}" for seed in range(n_seeds) for m in models] 

linestys = {'ppo': '--', 'a2c': '-'}
cols= {'ssp-xy': utils.blues[0], 'ssp-view':utils.purples[0], 'none': utils.oranges[0], 'xy': utils.reds[0]}  

linestys = {'image': '-', 'ssp-xy':'--', 'ssp-view':':'}
cols= {'cm': utils.reds[0], 'icm': utils.oranges[0],'lap': utils.blues[0],
        'aenc':utils.greens[0], 'latent': utils.yellows[0],
        'none': utils.purples[0],}


def match_dict(name, adict):
    for k in adict.keys():
        if k == name:
            return adict[k]
fig,ax =plt.subplots(1,1, figsize=(7,2))
model_names = set([m[:-2] for m in all_models])
frames, iqm_scores, iqm_cis, raw_results = utils.make_plots(model_names, env_name, 
                 linestys=dict(zip(model_names, [match_dict(m.split('_')[2], linestys) for m in model_names])),
                 cols=dict(zip(model_names, [match_dict(m.split('_')[3], cols) for m in model_names])), 
                 labels=dict(zip(model_names, [", ".join(m.split('_')[2:4]) for m in model_names])),
                 n_seeds=n_seeds,ax=ax,legend=True)

# moving avg causing drop at end
fig,axs =plt.subplots(4,1, figsize=(7,7))
for t,tt in enumerate(['cm','icm','latent','lap']):
    for mm in ['ssp-learn' 'ssp-xy','ssp-view','image']:
        axs[t].plot(frames, iqm_scores['MiniGrid-Empty-Random-8x8-v0_sr_' + mm + '_' + tt],label=mm)
    axs[t].legend()
    axs[t].set_title(tt)
    axs[t].set_ylim([0,1])

# import numpy as np
# np.savez('sr_8x8', env=env_name, n_seeds=n_seeds, model_names=model_names, frames=frames, iqm_scores=iqm_scores, iqm_cis=iqm_cis, raw_results=raw_results)

# fig=plt.figure(figsize=(8,4))
# linestys = {'image': '-', 'ssp-xy':'--', 'ssp-view':':'}
# cols= {'cm': utils.reds[0], 'icm': utils.oranges[0],'lap': utils.blues[0],
#        'aenc':utils.greens[0], 'latent': utils.yellows[0],
#        'none': utils.purples[0],}
# for i,model_name in enumerate(models):
#     model_dir = utils.get_model_dir(model_name)
#     data = pd.read_csv(model_dir + "/log.csv")
#     #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
#     input_type = model_name.split("_")[2]
#     learner = model_name.split("_")[3]
#     if model_name.split("_")[1]=='a2c':
#         sns.lineplot(x="frames", y='return_mean', label=", ".join(model_name.split("_")[1:]),
#                      data=data, alpha=0.8, linestyle=linestys[input_type], color='k')
#     else:
#         sns.lineplot(x="frames", y='return_mean', label=", ".join(model_name.split("_")[1:]),
#                      data=data, alpha=0.8, linestyle=linestys[input_type], color=cols[learner])
# plt.legend(frameon=True,edgecolor='white')
# plt.xlabel("Frames observed")
# plt.ylabel("Average Return")
# plt.title(env_name)
# # utils.save(fig,"figures/sr_" + env_name + ".pdf")
# utils.save(fig,"figures/rand6x6_sr_" + env_name + ".png")



# fig=plt.figure(figsize=(8,4))
# linestys = {'image': '-', 'ssp-xy':'--', 'ssp-view':':'}
# cols= {'cm': utils.reds[0], 'icm': utils.oranges[0],'lap': utils.blues[0],
#        'aenc':utils.greens[0], 'latent': utils.yellows[0],
#        'none': utils.purples[0],}
# for i,model_name in enumerate(models):
#     model_dir = utils.get_model_dir(model_name)
#     data = pd.read_csv(model_dir + "/log.csv")
#     #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
#     input_type = model_name.split("_")[2]
#     learner = model_name.split("_")[3]
#     data['return_rollingavg'] = data.return_mean.copy().rolling(50).mean()
#     if model_name.split("_")[1]=='a2c':
#         sns.lineplot(x="frames", y='return_rollingavg', label=", ".join(model_name.split("_")[1:]),
#                      data=data, alpha=0.8, linestyle=linestys[input_type], color='k')
#     else:
#         sns.lineplot(x="frames", y='return_rollingavg', label=", ".join(model_name.split("_")[1:]),
#                      data=data, alpha=0.8, linestyle=linestys[input_type], color=cols[learner])
# plt.legend(frameon=True,edgecolor='white')
# plt.xlabel("Frames observed")
# plt.ylabel("Average Return")
# plt.title(env_name)
# # utils.save(fig,"figures/sr_" + env_name + ".pdf")
# utils.save(fig,"figures/rand6x6_sr_" + env_name + "_ma50.png")


