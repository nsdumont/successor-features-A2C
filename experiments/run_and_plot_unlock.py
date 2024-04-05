import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys,os
os.chdir("..")
from train import run
import utils


models = ['plot_8x8_ppo_image', 'plot_8x8_ppo_ssp-view']
          #'plot_8x8_a2c_image','plot_8x8_a2c_xy', 'plot_8x8_a2c_ssp-xy',  'plot_8x8_a2c_ssp-view']
          # 'plot_8x8_sr_image','plot_8x8_sr_xy', 'plot_8x8_sr_ssp-xy', 'plot_8x8_sr_ssp-view',
          # 'plot_8x8_sr-ppo_image', 'plot_8x8_sr-ppo_xy', 'plot_8x8_sr-ppo_ssp-xy','plot_8x8_sr-ppo_ssp-view' ]
env_name = "MiniGrid-Unlock-v0"

  
for i,model_name in enumerate(models):
    print("Starting "+ model_name )
    algo = model_name.split("_")[2]
    wrapper = model_name.split("_")[3]

    if wrapper=='image':
        wrapper="none"
        input_type="image"
    else:
        input_type="flat"
    
    for seed in range(n_seeds):
        _model_name =  f"{env_name}_" + model_name + f"_{seed}" 
        
        model_dir = utils.get_model_dir(_model_name)
        if os.path.exists(model_dir):
            if replace_existing:
                for f in os.listdir(model_dir):
                    os.remove(os.path.join(model_dir, f))
                os.rmdir(model_dir)
            else:
                pass
            
        if wrapper=='ssp-view':
            wrapper_args={'ignore': ['WALL', 'FLOOR']}
        else:
            wrapper_args={}
        run(algo = algo, input=input_type, wrapper=wrapper, model = _model_name,seed=seed,
              env=env_name, frames=100000, entropy_coef=0.0005, verbose=False,
              wrapper_args=wrapper_args)
        
    print("Finsihed "+ model_name )

fig=plt.figure(figsize=(7.5,2.))
# linestys = {'ssp-xy': '-', 'image':'--'}
# cols= {'ppo': utils.reds[0], 'a2c': utils.blues[0],'sr': utils.oranges[0],'sr-ppo': utils.purples[0],}
linestys = {'ppo': '--', 'a2c':'-'}
cols= {'image': utils.reds[0], 'xy': utils.oranges[0],'ssp-xy': utils.blues[0],'ssp-view': utils.purples[0],}
for i,model_name in enumerate(models):
    df = pd.DataFrame()

    for seed in range(0,n_seeds):
        try:
            model_dir = utils.get_model_dir( f"{env_name}_" + model_name +  '_' + str(seed))
            data = pd.read_csv(model_dir + "/log.csv")
            #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
            data = data[pd.to_numeric(data['return_mean'], errors='coerce').notnull()]
            data['return_mean'] = pd.to_numeric(data['return_mean'])
            data['frames'] = pd.to_numeric(data['frames']).astype(float)
            df = df._append(data)
        except:
            pass
        
 
    #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
    input_type = model_name.split("_")[3]
    algo = model_name.split("_")[2]
    sns.lineplot(x="frames", y='return_mean', label=", ".join(model_name.split("_")[-2:]),errorbar='ci',
                 data=df, alpha=0.8, linestyle=linestys[algo], color=cols[input_type])
plt.legend(frameon=True,edgecolor='white')
plt.xlabel("Frames observed")
plt.ylabel("Average Return")
plt.title(env_name)
# utils.save(fig,"figures/" + env_name + ".pdf")
# utils.save(fig,"figures/" + env_name + ".png")


# fig=plt.figure(figsize=(7.5,2.))
# # linestys = {'ssp-xy': '-', 'image':'--'}
# # cols= {'ppo': utils.reds[0], 'a2c': utils.blues[0],'sr': utils.oranges[0],'sr-ppo': utils.purples[0],}
# linestys = {'ppo': '--', 'a2c':'-'}
# cols= {'image': utils.reds[0], 'xy': utils.oranges[0],'ssp-xy': utils.blues[0],'ssp-view': utils.purples[0],}
# for i,model_name in enumerate(models[4:]):
#     model_dir = utils.get_model_dir(model_name)
#     data = pd.read_csv(model_dir + "/log.csv")
#     #data['avg_return'] = data.return_mean.copy().rolling(100).mean()
#     input_type = model_name.split("_")[3]
#     algo = model_name.split("_")[2]
#     sns.lineplot(x="frames", y='return_mean', label=", ".join(model_name.split("_")[-2:]),
#                  data=data, alpha=0.8, linestyle=linestys[algo], color=cols[input_type])
# plt.legend(frameon=True,edgecolor='white')
# plt.xlabel("Frames observed")
# plt.ylabel("Average Return")
# plt.title(env_name)
# utils.save(fig,"figures/"+ env_name +"-a2c.pdf")
# utils.save(fig,"figures/" + env_name + "-a2c.png")