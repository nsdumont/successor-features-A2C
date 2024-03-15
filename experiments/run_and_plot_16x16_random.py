import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils

models = ['plot_16x16rand_ppo_image','plot_16x16rand_ppo_xy', 'plot_16x16rand_ppo_ssp-xy','plot_16x16rand_ppo_ssp-view',
          'plot_16x16rand_a2c_image','plot_16x16rand_a2c_xy', 'plot_16x16rand_a2c_ssp-xy',  'plot_16x16rand_a2c_ssp-view']
          # 'plot_8x8_sr_image','plot_8x8_sr_xy', 'plot_8x8_sr_ssp-xy', 'plot_8x8_sr_ssp-view',
          # 'plot_8x8_sr-ppo_image', 'plot_8x8_sr-ppo_xy', 'plot_8x8_sr-ppo_ssp-xy','plot_8x8_sr-ppo_ssp-view' ]

env_name = "MiniGrid-Empty-Random-16x16-v0"


for model_name in models:
    model_dir = utils.get_model_dir(model_name)
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, f))
        os.rmdir(model_dir)
                
for i,model_name in enumerate(models):
    print("Starting "+ model_name )
    algo = model_name.split("_")[2]
    input_type = model_name.split("_")[3]
    # if algo=='sr':
    #     other_args = '--entropy-coef 0.01 --entropy-decay 0.9 ' 
    #     feature_learn = 
    # else:
    entropy_coef=0.0005
    entropy_decay=0
    if input_type=='image':
        input_type="none"
        dissim_coef=0
        
    elif "ssp" in input_type:
        dissim_coef = 0.001
        if input_type=='ssp-xy':
            entropy_coef=0.001
            entropy_decay=0.1
    
    run(algo = algo, wrapper=input_type, model = model_name,dissim_coef=dissim_coef,
          env=env_name, frames=30000, entropy_coef=entropy_coef, entropy_decay=entropy_decay, erbose=False)
    print("Finsihed "+ model_name )


fig=plt.figure(figsize=(7.5,2.))
# linestys = {'ssp-xy': '-', 'image':'--'}
# cols= {'ppo': utils.reds[0], 'a2c': utils.blues[0],'sr': utils.oranges[0],'sr-ppo': utils.purples[0],}
linestys = {'ppo': '--', 'a2c':'-'}
cols= {'image': utils.reds[0], 'xy': utils.oranges[0],'ssp-xy': utils.blues[0],'ssp-view': utils.purples[0],}
for i,model_name in enumerate(models):
    model_dir = utils.get_model_dir(model_name)
    data = pd.read_csv(model_dir + "/log.csv")
    #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
    input_type = model_name.split("_")[3]
    algo = model_name.split("_")[2]
    sns.lineplot(x="frames", y='return_mean', label=", ".join(model_name.split("_")[-2:]),
                 data=data, alpha=0.8, linestyle=linestys[algo], color=cols[input_type])
plt.legend(frameon=True,edgecolor='white')
plt.xlabel("Frames observed")
plt.ylabel("Average Return")
plt.title(env_name)
utils.save(fig,"figures/" + env_name + ".pdf")
utils.save(fig,"figures/" + env_name + ".png")


fig=plt.figure(figsize=(7.5,2.))
# linestys = {'ssp-xy': '-', 'image':'--'}
# cols= {'ppo': utils.reds[0], 'a2c': utils.blues[0],'sr': utils.oranges[0],'sr-ppo': utils.purples[0],}
linestys = {'ppo': '--', 'a2c':'-'}
cols= {'image': utils.reds[0], 'xy': utils.oranges[0],'ssp-xy': utils.blues[0],'ssp-view': utils.purples[0],}
for i,model_name in enumerate(models[4:]):
    model_dir = utils.get_model_dir(model_name)
    data = pd.read_csv(model_dir + "/log.csv")
    #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
    input_type = model_name.split("_")[3]
    algo = model_name.split("_")[2]
    sns.lineplot(x="frames", y='return_mean', label=", ".join(model_name.split("_")[-2:]),
                 data=data, alpha=0.8, linestyle=linestys[algo], color=cols[input_type])
plt.legend(frameon=True,edgecolor='white')
plt.xlabel("Frames observed")
plt.ylabel("Average Return")
plt.title(env_name)
utils.save(fig,"figures/"+ env_name +"-a2c.pdf")
utils.save(fig,"figures/" + env_name + "-a2c.png")