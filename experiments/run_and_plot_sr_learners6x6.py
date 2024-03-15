import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils

models = ['plot_a2c_image_none','plot_a2c_ssp-xy_none',
    'plot_sr_image_cm','plot_sr_image_icm', 'plot_sr_image_aenc',
          'plot_sr_image_lap','plot_sr_image_latent',
          'plot_sr_ssp-xy_none',#'plot_sr_ssp-view_none',
          'plot_sr_ssp-xy_icm']


env_name = "MiniGrid-Empty-6x6-v0"

for model_name in models:
    model_dir = utils.get_model_dir(model_name)
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, f))
        os.rmdir(model_dir)
     
# runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 --plot True --normlize True --feature-learn cm --critic-hidden-size 256 --entropy-coef 0.01 --entropy-decay 0.9', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
   #runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 --plot True --normlize True --feature-learn cm --feature-hidden-size 128 --critic-hidden-size 256 --entropy-coef 0.01 --entropy-decay 0.9', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)        
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000  --wrapper ssp-xy --plot True --feature-learn icm --critic-hidden-size 128 --entropy-coef 0.01 --entropy-decay 0.9 --model plot_sr_ssp-xy_icm', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000  --wrapper ssp-xy --plot True --feature-learn none --critic-hidden-size 128 --entropy-coef 0.001 --entropy-decay 0.9  --model plot_sr_ssp-xy_none', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
for i,model_name in enumerate(models):
    print("Starting "+ model_name )
    algo = model_name.split("_")[1]
    wrapper = model_name.split("_")[2]
    feature_learn = model_name.split("_")[3]
    
    feature_hidden_size=128
    critic_hidden_size=256
    entropy_coef=0.001
    entropy_decay=0.1
    normalize=True
    if feature_learn=='none':
        normalize=False
        critic_hidden_size=128
    if algo=='a2c':
        critic_hidden_size=64  
        entropy_coef=0.0005
        entropy_decay=0.0
        normalize=False
    if wrapper=='image':
        wrapper="none"
    if ('cm' in feature_learn):#
        entropy_coef = 0.01
        entropy_decay= 0.9
    
    run(algo = algo, wrapper=wrapper, model = model_name, seed=1, normalize_embeddings=normalize,
          env=env_name, frames=50000, entropy_coef=entropy_coef, entropy_decay=entropy_decay,
          feature_hidden_size=feature_hidden_size,critic_hidden_size=critic_hidden_size,verbose=False)
    print("Finished "+ model_name )


fig=plt.figure(figsize=(8,4))
linestys = {'image': '-', 'ssp-xy':'--', 'ssp-view':':'}
cols= {'cm': utils.reds[0], 'icm': utils.oranges[0],'lap': utils.blues[0],
       'aenc':utils.greens[0], 'latent': utils.yellows[0],
       'none': utils.purples[0],}
for i,model_name in enumerate(models):
    model_dir = utils.get_model_dir(model_name)
    data = pd.read_csv(model_dir + "/log.csv")
    #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
    input_type = model_name.split("_")[2]
    learner = model_name.split("_")[3]
    data['return_rollingavg'] = data.return_mean.copy().rolling(50).mean()
    if model_name.split("_")[1]=='a2c':
        sns.lineplot(x="frames", y='return_rollingavg', label=", ".join(model_name.split("_")[1:]),
                     data=data, alpha=0.8, linestyle=linestys[input_type], color='k')
    else:
        sns.lineplot(x="frames", y='return_rollingavg', label=", ".join(model_name.split("_")[1:]),
                     data=data, alpha=0.8, linestyle=linestys[input_type], color=cols[learner])
plt.legend(frameon=True,edgecolor='white')
plt.xlabel("Frames observed")
plt.ylabel("Average Return")
plt.title(env_name)
# utils.save(fig,"figures/sr_" + env_name + ".pdf")
utils.save(fig,"figures/sr_" + env_name + "_ma50.png")


