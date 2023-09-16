import os
import utils


models = ['plot_5x5maze_ppo_flat', 'plot_5x5maze_ppo_ssp','plot_5x5maze_a2c_flat', 'plot_5x5maze_a2c_ssp',
          'plot_5x5maze_sr_flat', 'plot_5x5maze_sr_ssp']


models = ['plot_maze_ppo_flat', 'plot_maze_ppo_ssp',
          'plot_maze_a2c_flat',  'plot_maze_a2c_ssp', 
          'plot_maze_sr_flat', 'plot_maze_sr_ssp',
          'plot_maze_sr-ppo_flat', 'plot_maze_sr-ppo_ssp']

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
        other_args = ' --entropy-coef 0.1 --entropy-decay 0.95 ' 
    else:
        other_args = '--entropy-coef 0.0005 '
    if input_type=='ssp':
        input_type = 'ssp-xy'
        feature_learn = 'none'
        other_args = other_args + '--ssp-h 0.6 '
    else:
        feature_learn = 'curiosity'
    os.system("python train.py --algo " + algo + " --input " + input_type +
              " --feature-learn " + feature_learn + " --model " + model_name
          + " --env maze-sample-5x5 --frames 50000 " + other_args)



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plt.figure()
plt.title('maze-sample-5x5')
linestys = ['-','--','-','--','-','--','-','--']
for i,model_name in enumerate(models):
    model_dir = utils.get_model_dir(model_name)
    data = pd.read_csv(model_dir + "/log.csv")
    #data['avg_return'] = data.return_mean.copy().rolling(100).mean()

    sns.lineplot(x="frames", y='return_mean', label=model_name[10:], data=data, alpha=0.8, linestyle=linestys[i])
plt.legend()
plt.xlabel("Frames observed", size=14)
plt.ylabel("Average Return", size=14)