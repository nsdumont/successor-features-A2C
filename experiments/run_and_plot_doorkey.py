import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-DoorKey-8x8-v0 --frames 100000 --wrapper ssp-view --input flat --plot True --procs 1 --frames-per-proc 100 ', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)


envs = ['MiniGrid-DoorKey-16x16-v0']

models = ['a2c_ssp-view','a2c_xy']
for i,m in enumerate(models):
    models[i] = 'doorkey16x16_' + m

for model_name in models:
    model_dir = utils.get_model_dir(model_name)
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, f))
        os.rmdir(model_dir)

test_outs = []
for i, env in enumerate(envs):
    for model_name in models:
        print("Starting "+ model_name )
        algo = model_name.split("_")[1]
        wrapper = model_name.split("_")[2]
        
        dissim_coef = 0.0
        if wrapper=='image':
            wrapper="none"
            input="image"
        else:
            input="flat"
        if "ssp" in wrapper:
            dissim_coef = 0.001

           
           
        _, test_out= run(algo = algo, wrapper=wrapper, model = model_name,input=input,
               env=env, frames=50000, entropy_coef=0.001, dissim_coef=dissim_coef,
               verbose=False)
        test_outs.append(test_out)

models = ['a2c_image','a2c_ssp-xy',
          'a2c_ssp-view','a2c_xy']
for i,m in enumerate(models):
    models[i] = 'doorkey16x16_' + m

fig =plt.figure(figsize=(7,3))
linestys = {'ppo': '--', 'a2c':'-'}
cols= {'image': utils.reds[0], 'xy': utils.oranges[0],'ssp-xy': utils.blues[0],'ssp-view': utils.purples[0],}
for i,model_name in enumerate(models):
    model_dir = utils.get_model_dir(model_name)
    data = pd.read_csv(model_dir + "/log.csv")
    #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
    data = data[pd.to_numeric(data['return_mean'], errors='coerce').notnull()]
    data['return_mean'] = pd.to_numeric(data['return_mean'])
    data['frames'] = pd.to_numeric(data['frames'])
    input_type = model_name.split("_")[2]
    if input_type=='none':
        input_type ='image'
    algo = model_name.split("_")[1]
    data['return_rollingavg'] = data.return_mean.copy().rolling(50).mean()
    sns.lineplot(x="frames", y='return_mean', label=algo.upper() + "-" + input_type,
                 data=data, alpha=0.8)#, linestyle=linestys[input_type])#, color=cols[algo])

plt.title(envs[0])
plt.xlabel("Frames observed")
plt.ylabel("Average Return")
plt.legend(loc='lower right', facecolor='white', framealpha=1, frameon = True, edgecolor='none')
# utils.save(fig,'figures/' + env + '.pdf')