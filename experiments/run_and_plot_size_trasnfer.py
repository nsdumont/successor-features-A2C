import os
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# models = ['MiniGrid_size_sr_image','MiniGrid_size_sr_ssp']
models = ['MiniGrid_size_a2c_image','MiniGrid_size_a2c_ssp','MiniGrid_size_sr_image','MiniGrid_size_sr_ssp']

      #    'MiniGrid_size_a2c_ssp','MiniGrid_size_ppo_ssp','MiniGrid_size_sr_ssp']
#'MiniGrid_size_ppo_image'
for model_name in models:
    model_dir = utils.get_model_dir(model_name)
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, f))
        os.rmdir(model_dir)
                


for i,model_name in enumerate(models):
    for i,env in enumerate(["MiniGrid-Empty-6x6-v0", "MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-16x16-v0"]):
        algo = model_name.split("_")[2]
        input_type = model_name.split("_")[3]
        if algo=='sr':
            other_args = ' --entropy-coef 0.01 --entropy-decay 0.9 ' 
        else:
            other_args = ' --entropy-coef 0.0005 '
        if input_type=='ssp':
            input_type = 'ssp-xy'
            feature_learn = 'none'
            other_args = other_args + '--ssp-h ' + int(env.split("-")[2].split("x")[0])/6
        else:
            feature_learn = 'curiosity'
        os.system("python train.py --algo " + algo + " --input " + input_type +
                  " --feature-learn " + feature_learn + " --model " + model_name
              + " --env " + env + " --frames " + str(30000*(i+1)) + other_args)




plt.figure(figsize=(7,3))
linestys = {'ssp': '-', 'image':'--'}
cols= {'ppo': utils.reds[0], 'a2c': utils.oranges[0],'sr': utils.blues[0],'sr-ppo': utils.purples[0],}
for i,model_name in enumerate(models):
    model_dir = utils.get_model_dir(model_name)
    data = pd.read_csv(model_dir + "/log.csv")
    #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
    data = data[pd.to_numeric(data['return_mean'], errors='coerce').notnull()]
    data['return_mean'] = pd.to_numeric(data['return_mean'])
    data['frames'] = pd.to_numeric(data['frames'])
    input_type = model_name.split("_")[3]
    algo = model_name.split("_")[2]
    sns.lineplot(x="frames", y='return_mean', label=model_name[14:],
                 data=data, alpha=0.8, linestyle=linestys[input_type], color=cols[algo])

plt.xlabel("Frames observed")
plt.ylabel("Average Return")