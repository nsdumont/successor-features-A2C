import os
import utils


models = ['plot_5x5maze_ppo_image', 'plot_5x5maze_ppo_ssp','plot_5x5maze_a2c_image', 'plot_5x5maze_a2c_ssp',
          'plot_5x5maze_sr_image', 'plot_5x5maze_sr_ssp']

os.system("python train.py --algo ppo --input flat --feature-learn none --model " + models[0] 
          + " --env MiniWorld-MazeS3Fast-v0 --frames 30000 --entropy-coef 0.0005")
os.system("python train.py --algo ppo --input ssp-auto  --feature-learn none --model " +models[1] 
          + " --env MiniWorld-MazeS3Fast --frames 30000 --ssp-h 1 --entropy-coef 0.0005")
os.system("python train.py --algo a2c --input flat --feature-learn none --model " +models[2] 
          + " --env MiniWorld-MazeS3Fast --frames 30000 --entropy-coef 0.0005")
os.system("python train.py --algo a2c --input ssp-auto  --feature-learn none --model " +models[3] 
          + " --env MiniWorld-MazeS3Fast --frames 30000 --ssp-h 1 --entropy-coef 0.0005")
os.system("python train.py --algo sr --input flat --feature-learn curiosity --model " +models[4] 
          + " --env MiniWorld-MazeS3Fast --frames 30000  --entropy-coef 0.01 --entropy-decay 0.9")
os.system("python train.py --algo sr --input ssp-auto --feature-learn none --model " +models[5] 
          + " --env MiniWorld-MazeS3Fast --frames 30000  --ssp-h 1 --entropy-coef 0.01 --entropy-decay 0.9")


# runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniWorld-MazeS3Fast-v0 --frames 30000 --input ssp-miniworld-xy --procs 1 --feature-learn none --lr 0.02', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
# runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniWorld-MazeS3Fast-v0 --frames 30000 --input ssp-miniworld-xy --procs 1 --feature-learn none --lr 0.005', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
# runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniWorld-MazeS3Fast-v0 --frames 30000 --input ssp-miniworld-xy --procs 1 --feature-learn none --lr 0.0005', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)



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

    sns.lineplot(x="frames", y='return_mean', label=model_name[9:], data=data, alpha=0.8, linestyle=linestys[i])

plt.xlabel("Frames observed", size=14)
plt.ylabel("Average Return", size=14)