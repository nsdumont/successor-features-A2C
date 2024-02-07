import os
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-DoorKey-8x8-v0 --frames 100000 --wrapper ssp-view --input flat --plot True --procs 1 --frames-per-proc 100 ', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)



envs = [ 'MiniGrid-Empty-5x5-v0', 'MiniGrid-Dynamic-Obstacles-5x5-v0']
n_frames = [50000]*len(envs)
# envs = ['MiniGrid-Empty-8x8-v0', 'MiniGrid-Empty-16x16-v0', 'MiniGrid-FourRooms-v0',
#         'MiniWorld-TMazeLeft-v0', 'MiniWorld-YMazeLeft-v0', 
#         'MiniWorld-MazeS2-v0', 'MiniWorld-MazeS3-v0', 'MiniWorld-MazeS3Fast-v0']#,
      #  'maze-sample-5x5','maze-sample-10x10-v0','maze-sample-100x100-v0']
algos = ['a2c']
wrappers = ['none', 'ssp-view']

for i, env in enumerate(envs):
    models = []
    for algo in algos:

        for input_type in wrappers:
            if input_type=='none':
                in_ty = 'image'
                
            else:
                in_ty = 'flat'
                
            model_name = 'MiniGrid-DynObs' + '_' + algo + '_' + input_type
            models.append(model_name)
            if i==0:
                model_dir = utils.get_model_dir(model_name)
                if os.path.exists(model_dir):
                    for f in os.listdir(model_dir):
                        os.remove(os.path.join(model_dir, f))
                    os.rmdir(model_dir)
                load = False
            else:
                load = True

            os.system("python train.py --algo " + algo + " --wrapper " + input_type +
                      " --input " + in_ty + " --model " + model_name
                  + " --env  " + env + " --frames " + str(n_frames[i]*(i+1)) +
                  " --procs 1 --frames-per-proc 100 --plot False --entropy-coef 0.001 --load-optimizer-state " + str(load) )
                
            


fig =plt.figure(figsize=(7,3))
linestys = {'ssp': '-', 'image':'--'}
cols= {'ppo': utils.reds[0], 'a2c': utils.oranges[0],'sr': utils.blues[0],'sr-ppo': utils.purples[0],}
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
    sns.lineplot(x="frames", y='return_mean', label=algo.upper() + "-" + input_type,
                 data=data, alpha=0.8)#, linestyle=linestys[input_type])#, color=cols[algo])
for i in range(len(envs)):
    plt.axvline(x = n_frames[i]*(i+1), color = 'k')
plt.title("MiniGrid-DoorKey: 5x5, 6x6, 8x8, 16x16")
plt.xlabel("Frames observed")
plt.ylabel("Average Return")
plt.legend(loc='lower right', facecolor='white', framealpha=1, frameon = True, edgecolor='none')
utils.save(fig,'MiniGrid-DynObs' + '.pdf')