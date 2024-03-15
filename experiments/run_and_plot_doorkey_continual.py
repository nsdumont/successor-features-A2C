import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-DoorKey-8x8-v0 --frames 100000 --wrapper ssp-view --input flat --plot True --procs 1 --frames-per-proc 100 ', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)


#
envs = ['MiniGrid-DoorKey-5x5-v0',  'MiniGrid-DoorKey-6x6-v0', 'MiniGrid-DoorKey-8x8-v0',
        'MiniGrid-DoorKey-16x16-v0']
n_frames = [50000]*len(envs)
# envs = ['MiniGrid-Empty-8x8-v0', 'MiniGrid-Empty-16x16-v0', 'MiniGrid-FourRooms-v0',
#         'MiniWorld-TMazeLeft-v0', 'MiniWorld-YMazeLeft-v0', 
#         'MiniWorld-MazeS2-v0', 'MiniWorld-MazeS3-v0', 'MiniWorld-MazeS3Fast-v0']#,
      #  'maze-sample-5x5','maze-sample-10x10-v0','maze-sample-100x100-v0']
algos = ['a2c']
wrappers = ['image', 'xy', 'ssp-view']
test_outs = []
for i, env in enumerate(envs):
    models = []
    for algo in algos:

        for input_type in wrappers:
            
                
            model_name = 'MiniGrid-DoorKey2' + '_' + algo + '_' + input_type
            
            if input_type=='image':
                input_type='none'
            if 'ssp' in input_type:
                procs=2
                frames_per_proc=25
            else:
                procs=5
                frames_per_proc=5
            
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
                
            _, test_out= run(algo = algo, wrapper=input_type, model = model_name,
                procs=procs,frames_per_proc=frames_per_proc,
                  env=env, frames=n_frames[i]*(i+1), entropy_coef=0.001, 
                  load_optimizer_state= load,
                  verbose=False)
            test_outs.append(test_out)
            # os.system("python train.py --algo " + algo + " --wrapper " + input_type +
            #            + " --model " + model_name
            #       + " --env  " + env + " --frames " + str(n_frames[i]*(i+1)) +
            #       " --procs 1 --frames-per-proc 100 --plot False --entropy-coef 0.001 --load-optimizer-state " + str(load) )
                
            


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
    sns.lineplot(x="frames", y='return_mean', label=algo.upper() + "-" + input_type,
                 data=data, alpha=0.8)#, linestyle=linestys[input_type])#, color=cols[algo])
for i in range(len(envs)):
    plt.axvline(x = n_frames[i]*(i+1), color = 'k')
plt.title("MiniGrid-DoorKey: 5x5, 6x6, 8x8, 16x16")
plt.xlabel("Frames observed")
plt.ylabel("Average Return")
plt.legend(loc='lower right', facecolor='white', framealpha=1, frameon = True, edgecolor='none')
# utils.save(fig,'figures/' + env + '.pdf')