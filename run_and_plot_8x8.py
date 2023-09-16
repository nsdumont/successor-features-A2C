import os
import utils

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


models = ['plot_8x8_ppo_image', 'plot_8x8_ppo_ssp','plot_8x8_a2c_image', 
          'plot_8x8_a2c_ssp', 'plot_8x8_sr_image', 'plot_8x8_sr_ssp',
          'plot_8x8_sr-ppo_image', 'plot_8x8_sr-ppo_ssp']

for model_name in models:
    model_dir = utils.get_model_dir(model_name)
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, f))
        os.rmdir(model_dir)
                
for i,model_name in enumerate(models):
    algo = model_name.split("_")[2]
    input_type = model_name.split("_")[3]
    if algo=='sr':
        other_args = '--entropy-coef 0.01 --entropy-decay 0.9 ' 
    else:
        other_args = '--entropy-coef 0.0005 '
    if input_type=='ssp':
        input_type = 'ssp-xy'
        feature_learn = 'none'
        other_args = other_args + '--ssp-h 1 '
    else:
        feature_learn = 'curiosity'
    os.system("python train.py --algo " + algo + " --input " + input_type +
              " --feature-learn " + feature_learn + " --model " + model_name
          + " --env MiniGrid-Empty-8x8-v0 --frames 50000 " + other_args)



plt.figure(figsize=(7,3))
linestys = {'ssp': '-', 'image':'--'}
cols= {'ppo': utils.reds[0], 'a2c': utils.oranges[0],'sr': utils.blues[0],'sr-ppo': utils.purples[0],}
for i,model_name in enumerate(models):
    model_dir = utils.get_model_dir(model_name)
    data = pd.read_csv(model_dir + "/log.csv")
    #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
    input_type = model_name.split("_")[3]
    algo = model_name.split("_")[2]
    sns.lineplot(x="frames", y='return_mean', label=model_name[9:],
                 data=data, alpha=0.8, linestyle=linestys[input_type], color=cols[algo])

plt.xlabel("Frames observed")
plt.ylabel("Average Return")