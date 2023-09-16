import optuna
import argparse
import time
import datetime
import torch
import utils
import copy
import numpy as np
import yaml
import minigrid
# import miniworld
import pandas as pd
import tensorboardX
import sys
import seaborn as sns
from models.model import ACModel

#from algos.sr_a2c import SRAlgo
from algos.a2c import A2CAlgo
from algos.ppo import PPOAlgo
from models.model_SR import SRModel

#python train.py --algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 --input image --feature-learn curiosity --lr_sr 0.01

#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 --input ssp-minigrid-xy --feature-learn none --lr_sr 0.01 --ssp-h 1', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniWorld-TMazeLeft-v0 --frames 50000 --input ssp-miniworld-xy --feature-learn none --lr_sr 0.01 --ssp-h 1 --procs 1', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
# Parse arguments
parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo | sr | sr-ppo (REQUIRED)")
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
parser.add_argument("--procs", type=int, default=5,
                    help="number of processes (default: 5)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--load-optimizer-state", type=int, default=1,
                    help="If True and a logs for this model (defined by model arg) exist then load the optimizer info from last run. Otherwise do not.")

parser.add_argument("--target-update", type=int, default=10,
                    help="how often to update the target network") # right now only set up for sr algo


## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO & reward function learning in SR (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")

# parser.add_argument("--lr", type=float, default=0.001,
#                     help="learning rate for all (default: 0.001)")
# parser.add_argument("--lr_f", type=float, default=None,
#                     help="learning rate for feature (default: 0.001)")
# parser.add_argument("--lr_a", type=float, default=None,
#                     help="learning rate for actor (default: 0.001)")
# parser.add_argument("--lr_sr", type=float, default=None,
#                     help="learning rate for SR (default: 0.001)")
# parser.add_argument("--lr_r", type=float, default=None,
#                     help="learning rate for reward (default: 0.00001)")

parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.0005,
                    help="entropy term coefficient (default: 0.0005)")
parser.add_argument("--memory-cap", type=int, default=100000,
                    help=" (default: 100000)")


parser.add_argument("--recon-loss-coef", type=float, default=0.1,
                    help="recontruction term coefficient (default: 0.1)")
parser.add_argument("--norm-loss-coef", type=float, default=1,
                    help="norm loss term coefficient (default: 1)")

parser.add_argument("--input", type=str, default="image",
                    help="format of input:  image | flat | ssp (default: image)")
parser.add_argument("--feature-learn", type=str, default="curiosity",
                    help="method for feature learning:  curiosity | reconstruction | none  (default: curiosity)")
parser.add_argument("--ssp-dim", type=int, default=151,
                    help="Dim of spp (default: 151)")
parser.add_argument("--ssp-h", type=float, default=1.,
                    help="Length scale of spp representation (default: 1)")

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

parser.add_argument("--env-args", type=yaml.load, default={'render_mode': 'rgb_array'},
                    help="")

args = parser.parse_args()


args.mem = args.recurrence > 1
#6x6
# Trial 13 finished with value: 0.812312498986721 and parameters: {'lr_a': 0.00450539479080254, 'lr_sr': 0.03739130665075426, 'lr_r': 0.0021586597837410116, 'batch_size_power': 7, 'target_update': 15, 'gae_lambda': 0.9549719430245037, 'value_loss_coef': 1.0621771953288506}. Best is trial 13 with value: 0.812312498986721.
# Trial 13: average return 0.812

#8x8
# Trial 9 finished with value: 0.7907499974966049 and parameters: {'lr_a': 0.016665010496731886, 'lr_sr': 0.005875331869782301, 'lr_r': 0.03873715904873485}. Best is trial 9 with value: 0.7907499974966049.
# Trial 9: average return 0.791

def objective(trial):
    args.lr_a = trial.suggest_float('lr_a', 0.0005, 0.05, log=True)
    args.lr_sr = trial.suggest_float('lr_sr', 0.005, 0.05, log=True)
    args.lr_f = 0.001
    args.lr_r = trial.suggest_float('lr_r', 0.0005, 0.05, log=True)
    args.batch_size = 2**trial.suggest_int('batch_size_power', 7, 9)
    args.target_update = trial.suggest_int('target_update', 5, 20)
    args.gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1)
    args.entropy_coef =0.0005# trial.suggest_float('entropy_coef', 1e-6, 1e-2, log=True)
    args.value_loss_coef = trial.suggest_float('value_loss_coef',0.1,2)
    
    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"optuna_{args.algo}_{args.env}_{trial.number}_{date}"
    model_name = default_model_name
    model_dir = utils.get_model_dir(model_name)
    
    # Load loggers and Tensorboard writer
    
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    

    # Set seed for all randomness sources
    
    utils.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load environments
    envs = []
    if args.input =='ssp-auto':
        from wrappers import SSPEnvWrapper
        for i in range(args.procs):
            envs.append( SSPEnvWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed,
                                auto_convert_spaces = True, shape_out = args.ssp_dim, length_scale=args.ssp_h,
                                decoder_method = 'from-set'))
    elif args.input =='ssp-minigrid-xy':
        from wrappers import SSPMiniGridXYWrapper
        for i in range(args.procs):
            envs.append( SSPMiniGridXYWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed,
                                 shape_out = args.ssp_dim,  length_scale=args.ssp_h, decoder_method = 'from-set'))
    elif args.input =='ssp-minigrid-view':
        from wrappers import SSPMiniGridViewWrapper
        for i in range(args.procs):
            envs.append( SSPMiniGridViewWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed,
                                 shape_out = args.ssp_dim, length_scale=args.ssp_h, decoder_method = 'from-set') )    
    elif args.input =='ssp-miniworld-xy':
        from wrappers import SSPMiniWorldXYWrapper
        for i in range(args.procs):
            envs.append( SSPMiniWorldXYWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed,
                                 shape_out = args.ssp_dim,  length_scale=args.ssp_h, decoder_method = 'from-set'))
    else:
        for i in range(args.procs):
            envs.append(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args))
    
    
    # Load training status
    status = {"num_frames": 0, "update": 0}
    
    # Load observations preprocessor
    
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)

    # Load model
    is_sr = (args.algo == "sr") or (args.algo == "sr-ppo")
    if is_sr:
        model = SRModel(obs_space, envs[0].action_space, args.mem, args.text, args.input, args.feature_learn,obs_space_sampler=envs[0].observation_space)
    else:
        model = ACModel(obs_space, envs[0].action_space, args.mem, args.text, args.input,obs_space_sampler=envs[0].observation_space)
    target = copy.deepcopy(model)

    model.to(device)
    target.to(device)
    
    # Load algo
    #reshape_reward = lambda o,a,r,d: -0.1 if r==0 else 10
    if args.algo == "a2c":
        algo = A2CAlgo(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = PPOAlgo(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    elif args.algo == "sr":
        from algos.sr_a2c import SRAlgo
        algo = SRAlgo(envs, model, target, args.feature_learn, device, args.frames_per_proc, args.discount, args.lr_a,args.lr_f,args.lr_sr,args.lr_r, args.gae_lambda,
                                args.entropy_coef,args.recon_loss_coef,args.norm_loss_coef,
                                args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, args.memory_cap, args.batch_size, preprocess_obss)
    
    elif args.algo == "sr-ppo":
        from algos.sr_ppo import SRPPOAlgo
        algo = SRPPOAlgo(envs, model, target, args.feature_learn, device, args.frames_per_proc, args.discount, args.lr_a,args.lr_f,args.lr_sr,args.lr_r, args.gae_lambda,
                                args.entropy_coef,args.recon_loss_coef,args.norm_loss_coef,
                                args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, args.memory_cap, args.epochs, args.batch_size, args.clip_eps , preprocess_obss,None)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))
    

    
    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    first_line=num_frames==0
    
    returns = []
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
            
            if is_sr:
                header += ["entropy", "policy_loss", "sr_loss",
                           "feature_loss","reward_loss","grad_norm", ]
                data += [logs["entropy"],  logs["policy_loss"], logs["sr_loss"],
                         logs["feature_loss"], logs["reward_loss"], logs["grad_norm"]]
        
            else:
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        
            
            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()
    
            if first_line:
                csv_logger.writerow(header)
                first_line = False
            csv_logger.writerow(data)
            csv_file.flush()
    
            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)
    
            returns.append(rreturn_per_episode['mean'])
    
        # Save status
    
        if args.save_interval > 0 and update % args.save_interval == 0:
            if is_sr and args.feature_learn!="none":
                status = {"num_frames": num_frames, "update": update,
                      "model_state": model.state_dict(),"reward_optimizer_state": algo.reward_optimizer.state_dict(),
                      "sr_optimizer_state": algo.sr_optimizer.state_dict(),# "actor_optimizer_state": algo.actor_optimizer.state_dict(),
                      "feature_optimizer_state": algo.feature_optimizer.state_dict()}
            elif is_sr and args.feature_learn=="none":
                status = {"num_frames": num_frames, "update": update,
                      "model_state": model.state_dict(),"reward_optimizer_state": algo.reward_optimizer.state_dict(),
                      "sr_optimizer_state": algo.sr_optimizer.state_dict()}    
            else:
                status = {"num_frames": num_frames, "update": update,
                      "model_state": model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
   
    data = pd.read_csv(model_dir + "/log.csv")
    sns.lineplot(x="frames", y='return_mean', data=data)
    final_avg_return =np.mean(data['return_mean'].values[-20:])
    print("Trial {}: average return {:.3f}".format(trial.number, final_avg_return))
    return final_avg_return

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

# Trial 19 finished with value: 0.3142499992251396 and parameters: {'lr_a': 0.0004695299249617771, 'lr_sr': 0.000852569887080032, 'lr_r': 0.0002102061501175344, 'batch_size_power': 6, 'target_update': 18, 'gae_lambda': 0.6046068067493853, 'entropy_coef': 0.0007988198890669333, 'value_loss_coef': 1.7377793516797575}. Best is trial 19 with value: 0.3142499992251396.
# Trial 60 finished with value: 0.7748242259025575 and parameters: {'lr_a': 0.002141318610814453, 'lr_sr': 0.00538969226362907, 'lr_r': 0.003485252952901969, 'batch_size_power': 9, 'target_update': 6, 'gae_lambda': 0.9716815154807764, 'value_loss_coef': 1.449830512228507}. Best is trial 60 with value: 0.7748242259025575.

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# data = pd.read_csv(model_dir + "/log.csv")
# sns.lineplot(x="frames", y='return_mean', data=data)
# plt.title('Return')
# plt.figure()
# sns.lineplot(x="frames", y='entropy', data=data)
# plt.title('Entropy')
# plt.figure()
# sns.lineplot(x="frames", y='policy_loss', data=data)
# plt.title('Policy loss')
# plt.figure()
# sns.lineplot(x="frames", y='sr_loss', data=data)
# plt.title('SR loss')
# plt.figure()
# sns.lineplot(x="frames", y='feature_loss', data=data)
# plt.title('Feature loss')
# plt.figure()
# sns.lineplot(x="frames", y='reward_loss', data=data)
# plt.title('Reward loss')
# plt.figure()
# sns.lineplot(x="frames", y='grad_norm', data=data)
# plt.title('Grad norm')


# model.reward.weight

# r_preds = []
# rs = []
# obs,_=envs[0].reset()
# plt.figure()
# plt.imshow(envs[0].render())
# for i in range(20):
#     preprocessed_obs = algo.preprocess_obss([obs], device=algo.device)
#     dist, value, embedding, _, successor, r_pred, _ = model(preprocessed_obs)
#     action = dist.sample().detach()
#     obs, reward, terminated, truncated, _ = envs[0].step(action.cpu().numpy())
#     r_preds.append(r_pred)
#     rs.append(reward)
#     plt.figure()
#     plt.imshow(envs[0].render())
    
    
    
# r_preds = []
# rs = []
# obs,_=envs[0].reset()
# for i in range(1000):
#     preprocessed_obs = algo.preprocess_obss([obs], device=algo.device)
#     dist, value, embedding, _, successor, r_pred, _ = model(preprocessed_obs)
#     action = dist.sample().detach()
#     obs, reward, terminated, truncated, _ = envs[0].step(action.cpu().numpy())
#     r_preds.append(r_pred.detach().cpu().numpy())
#     rs.append(reward)
#     if terminated or truncated:
#         obs,_=envs[0].reset()
    
# idx = np.where([r>0 for r in rs])[0][0]
# print(rs[idx-5:idx+5])
# print(r_preds[idx-5:idx+5])