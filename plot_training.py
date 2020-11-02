import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils

sns.set()
sns.set_context("paper")


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--save", type=str, default=None,
                    help="store output with the given filename")
args = parser.parse_args()

model = args.model

model_dir = utils.get_model_dir(model)
data = pd.read_csv(model_dir + "/log.csv")#,skiprows=lambda x: ((x != 0) and not x % 2 and x<19 ) or (x > 100))
g = sns.lineplot(x="frames", y='return_mean', data=data)
#data['return_mean'].rolling(100).mean().plot()
#plt.axhline(90,color="orange",linewidth =1.5)
g.set(xlabel = 'Environment Observations', ylabel = 'Mean Return')


if args.save is not None:
    plt.savefig(args.save)


#debugfile('/home/ns2dumon/Documents/GitHub/successor-features-A2C/train.py',args=' --algo sr --env MiniGrid-Empty-6x6-v0 --input image  --frames 100000 --seed 1  --lr 0.001  --recon-loss-coef 5 --entropy-coef 0.005 --feature-learn curiosity --target-update 100', wdir ='/home/ns2dumon/Documents/GitHub/successor-features-A2C')

sns.lineplot(x="update", y='A_mse', data=data)
plt.ylim([0,1])