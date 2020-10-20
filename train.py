import argparse
import time
import datetime
import torch
import utils
import copy
import numpy as np
import yaml

import tensorboardX
import sys

from models.model import ACModel

from algos.sr_a2c import SRAlgo
from algos.a2c import A2CAlgo
from algos.ppo import PPOAlgo
from models.model_SR import SRModel

#runfile('/home/ns2dumon/Documents/GitHub/successor-features-A2C/train.py',args=' --algo sr --env MiniGrid-Empty-6x6-v0 --frames 100000 --input image --feature-learn curiosity --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 100 ', wdir ='/home/ns2dumon/Documents/GitHub/successor-features-A2C'

#
#runfile('/home/ns2dumon/Documents/GitHub/successor-features-A2C/train.py',args=' --algo sr --env MiniGrid-Empty-6x6-v0 --frames 100000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10', wdir ='/home/ns2dumon/Documents/GitHub/successor-features-A2C')



# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo | sr (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 5)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--load-optimizer-state", type=bool, default=True,
                    help="If True and a logs for this model (defined by model arg) exist then load the optimizer info from last run. Otherwise do not.")

parser.add_argument("--target-update", type=int, default=100,
                    help="how often to update the target network") # right now only set up for sr algo


## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=300,
                    help="batch size for PPO & reward function learning in SR (default: 300)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")

parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate for all (default: 0.001)")
parser.add_argument("--lr_f", type=float, default=None,
                    help="learning rate for feature (default: 0.001)")
parser.add_argument("--lr_a", type=float, default=None,
                    help="learning rate for actor (default: 0.001)")
parser.add_argument("--lr_sr", type=float, default=None,
                    help="learning rate for SR (default: 0.001)")
parser.add_argument("--lr_r", type=float, default=None,
                    help="learning rate for reward (default: 0.001/30)")

parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.005,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--memory-cap", type=int, default=10000,
                    help=" (default: 10000)")



parser.add_argument("--sr-loss-coef", type=float, default=1,
                    help="sr loss term coefficient (default: 0.5)")
parser.add_argument("--policy-loss-coef", type=float, default=1,
                    help="policy loss term coefficient (default: 1)")
parser.add_argument("--recon-loss-coef", type=float, default=2,
                    help="recontruction term coefficient (default: 2)")
parser.add_argument("--reward-loss-coef", type=float, default=1,
                    help="reward loss term coefficient (default: 1)")
parser.add_argument("--norm-loss-coef", type=float, default=1,
                    help="norm loss term coefficient (default: 1)")
#parser.add_argument("--rank-loss-coef", type=float, default=0,
#                    help="rank loss term coefficient (default: 1)")
#parser.add_argument("--kl-loss-coef", type=float, default=0.1,
#                    help="kl loss term coefficient (default: 1)")

parser.add_argument("--input", type=str, default="image",
                    help="format of input:  image | flat  (default: image)")
parser.add_argument("--feature-learn", type=str, default="curiosity",
                    help="method for feature learning:  curiosity | reconstruction  (default: curiosity)")


parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=10,
                    help="maximum norm of gradient (default: 10)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

parser.add_argument("--env-args", type=yaml.load, default={},
                    help="")

args = parser.parse_args()

args.mem = args.recurrence > 1


args.lr_a = args.lr_a or args.lr
args.lr_sr = args.lr_sr or args.lr
args.lr_f = args.lr_f or args.lr
args.lr_r = args.lr_r or args.lr/30


# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments

envs = []
if args.input =='ssp':
    import nengo_ssp as ssp
    from gym_minigrid.wrappers import SSPWrapper

    X,Y,_ = ssp.HexagonalBasis(10,10)
    d = len(X.v)
    for i in range(args.procs):
        envs.append(SSPWrapper( utils.make_env(args.env,args.env_args, args.seed + 10000 * i),d,X,Y,delta=2, rng=np.random.RandomState(args.seed)))
else: # flat (won't work for minigrid envs but will for others) or image
    for i in range(args.procs):
        envs.append(utils.make_env(args.env,args.env_args, args.seed + 10000 * i))

txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model
if args.algo == "sr":
    model = SRModel(obs_space, envs[0].action_space, device, args.mem, args.text, args.input, args.feature_learn)
else:
    model = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
target = copy.deepcopy(model)
if "model_state" in status:
    model.load_state_dict(status["model_state"])
if "target_state" in status:
    target.load_state_dict(status["target_state"])
model.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(model))

# Load algo

reshape_reward = lambda o,a,r,d: -1 if r==0 else 10
if args.algo == "a2c":
    algo = A2CAlgo(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss)
elif args.algo == "ppo":
    algo = PPOAlgo(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
elif args.algo == "sr":
    reshape_reward = lambda o,a,r,d: -1 if r==0 else 10
    algo = SRAlgo(envs, model, target, args.feature_learn, device, args.frames_per_proc, args.discount, args.lr_a,args.lr_f,args.lr_sr,args.lr_r, args.gae_lambda,
                            args.entropy_coef, args.sr_loss_coef, args.policy_loss_coef,args.recon_loss_coef,args.reward_loss_coef,args.norm_loss_coef,
                            args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, args.memory_cap, args.batch_size, preprocess_obss,reshape_reward)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if ("optimizer_state" in status) and args.load_optimizer_state:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()
first_line=True

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1
    
    # Update target
    
    if update % args.target_update == 0:
        target.load_state_dict(model.state_dict())

    # Print logs
    

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        if args.algo == "sr":
            header += ["entropy", "value_loss", "policy_loss", "sr_loss",
                       "reconstruction_loss","reward_loss","norm_loss", "grad_norm", "A_mse",]
            data += [logs["entropy"], logs["value_loss"], logs["policy_loss"], logs["sr_loss"],
                     logs["reconstruction_loss"], logs["reward_loss"], logs["norm_loss"],logs["grad_norm"],logs["A_mse"]]
    
           # txt_logger.info(
            #    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | srL {:.3f} | reconL {:.3f} | rL {:.3f} | ∇ {:.3f}"
             #   .format(*data))
            txt_logger.info("Frames {}, Mean reward {:.3f}, Value loss {:.3f}, Policy loss {:.3f}, SR loss {:.3f}, Reward loss {:.3f}, Reconstruction loss {:.3f}".format(num_frames, rreturn_per_episode['mean'], logs["value_loss"], logs["policy_loss"], logs["sr_loss"],logs["reward_loss"],logs["reconstruction_loss"]))
        else:
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
    
            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if first_line:
            csv_logger.writerow(header)
        first_line = False
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)


    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        if args.algo == "sr":
            status = {"num_frames": num_frames, "update": update,
                  "model_state": model.state_dict(),
                  "sr_optimizer_state": algo.sr_optimizer.state_dict(),# "actor_optimizer_state": algo.actor_optimizer.state_dict(),
                  "reward_optimizer_state": algo.reward_optimizer.state_dict(), "feature_optimizer_state": algo.feature_optimizer.state_dict()}
        else:
            status = {"num_frames": num_frames, "update": update,
                  "model_state": model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_csv(model_dir + "/log.csv")
sns.lineplot(x="frames", y='return_mean', data=data)