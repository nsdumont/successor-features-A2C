import os
import utils

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


envs = ['MiniWorld-MazeS2-v0', 'MiniWorld-MazeS3-v0']
# envs = ['MiniGrid-Empty-8x8-v0', 'MiniGrid-Empty-16x16-v0', 'MiniGrid-FourRooms-v0',
#         'MiniWorld-TMazeLeft-v0', 'MiniWorld-YMazeLeft-v0', 
#         'MiniWorld-MazeS2-v0', 'MiniWorld-MazeS3-v0', 'MiniWorld-MazeS3Fast-v0']#,
      #  'maze-sample-5x5','maze-sample-10x10-v0','maze-sample-100x100-v0']
algos = ['ppo','a2c']
all_inputs = ['image','flat', 'ssp']
maze_inputs = ['flat', 'ssp']

all_models = []

for env in envs:
    models = []
    if env[:4] == 'maze':
        inputs = maze_inputs
    else:
        inputs = all_inputs
    for algo in algos:

        for input_type in inputs:
            if algo=='sr':
                other_args = '--entropy-coef 0.01 --entropy-decay 0.9 ' 
            else:
                other_args = ' --entropy-coef 0.0005 '
                
            if input_type=='ssp':
                input_type = 'ssp-xy'
                feature_learn = 'none'
                other_args = other_args + ' --ssp-h 1 '
            else:
                feature_learn = 'curiosity'
            
            if env[:9] == 'MiniWorld':
                other_args = other_args + ' --procs 1 '
                
            model_name = env + '_' + algo + '_' + input_type
            models.append(model_name)
            model_dir = utils.get_model_dir(model_name)
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    os.remove(os.path.join(model_dir, f))
                os.rmdir(model_dir)

            os.system("python train.py --algo " + algo + " --input " + input_type +
                      " --feature-learn " + feature_learn + " --model " + model_name
                  + " --env  " + env + " --frames 50000 " + other_args)
    all_models.append(models)



linestys = {'ssp-xy': '-', 'image':'--', 'flat': '-.'}
cols= {'ppo': utils.reds[0], 'a2c': utils.oranges[0],'sr': utils.blues[0],'sr-ppo': utils.purples[0],}
for j,env in enumerate(envs):
    plt.figure(figsize=(7,3))
    plt.xlabel("Frames observed")
    plt.ylabel("Average Return")
    plt.title(env)
    for i,model_name in enumerate(all_models[j]):
        model_dir = utils.get_model_dir(model_name)
        data = pd.read_csv(model_dir + "/log.csv")
        data['avg_return'] = data.return_mean.copy().rolling(100).mean()
        input_type = model_name.split("_")[2]
        algo = model_name.split("_")[1]
        sns.lineplot(x="frames", y='avg_return', label=model_name[9:],
                     data=data, alpha=0.8, linestyle=linestys[input_type], color=cols[algo])

