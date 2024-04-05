import argparse
import time
import datetime
import torch
import utils
import copy
import numpy as np
import yaml
import json


from torch.utils.tensorboard import SummaryWriter
import sys

from models.model import ACModel

#from algos.sr_a2c import SRAlgo
from algos.a2c import A2CAlgo
from algos.ppo import PPOAlgo
from models.sr_model import SRModel

#python train.py --algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 -a2c --lr_sr 0.01
#python train.py --algo a2c --env MiniGrid-Empty-6x6-v0 --frames 30000
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-Empty-6x6-v0 --frames 30000', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-Empty-6x6-v0 --frames 30000 --wrapper ssp-xy --plot True', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-Empty-6x6-v0 --frames 30000 --wrapper xy --plot True', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-Empty-6x6-v0 --frames 30000 --wrapper one-hot --plot True', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-Empty-6x6-v0 --frames 30000 --wrapper xy --input ssp --plot True', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)

#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 30000 --wrapper ssp-xy --input none --plot True --feature-learn none', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 30000 --wrapper xy --input ssp --plot True --feature-learn combined', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 30000 --wrapper none --input image --plot True --feature-learn curiosity', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)

#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 --input ssp-xy --feature-learn none --lr_sr 0.01 --ssp-h 1', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniWorld-TMazeLeft-v0 --frames 50000 --input ssp-xy --feature-learn none --lr_sr 0.01 --ssp-h 1 --procs 1', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env maze-sample-5x5 --frames 30000 --input flat --procs 5 ', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)


#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env maze-random-7x7 --frames 30000 --wrapper ssp-auto --plot True --n_test_episodes 0 --ssp-h 1 --lr 0.001 --entropy-decay 1e-5 --gae-lambda 1 --optim-eps 1e-9 --entropy-coef 0.0002 ', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)


def run(args=None,**kwargs): 
    if args is None:
        if "config" in kwargs:
            configfilename = kwargs['config']
        else:
            configfilename = 'default_config.yml'
        with open(configfilename, 'r') as f: # load the defaults
            config = yaml.load(f, Loader=yaml.FullLoader)
        config.update(kwargs) # 
        args = argparse.Namespace(**config)
    
    if not hasattr(args, "wrapper_args"):
        args.wrapper_args = {}
    
    args.mem = args.recurrence > 1
    args.lr_a = args.lr_a or args.lr
    args.lr_sr = args.lr_sr or args.lr
    args.lr_f = args.lr_f or args.lr
    args.lr_r = args.lr_r or args.lr
    
    # if args.input[:3]=='ssp':
    #     args.feature_learn = 'none'
    
    # Set run dir
    
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"
    
    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)
    
    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir, args.verbose)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = SummaryWriter(model_dir)
    
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
    if "MiniGrid" in args.env:
        import minigrid
    elif "MiniWorl" in args.env:
        import miniworld
    elif "maze" in args.env:
        import gym_maze
        
    if args.wrapper=='none':
        for i in range(args.procs):
            envs.append(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args))
    elif args.wrapper =='ssp-auto':
        from wrappers import SSPEnvWrapper
        for i in range(args.procs):
            envs.append( SSPEnvWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed,
                                auto_convert_obs_space = True,auto_convert_action_space=False, shape_out = args.ssp_dim, length_scale=args.ssp_h,
                                decoder_method = 'from-set', **args.wrapper_args))
    elif ('MiniGrid' in args.env) or ('BabyAI' in args.env):
        if (args.wrapper =='xy'):
            from wrappers import MiniGridXYWrapper
            for i in range(args.procs):
                envs.append( MiniGridXYWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args),
                                               seed=args.seed, **args.wrapper_args))
        elif (args.wrapper =='one-hot'):
            from wrappers import MiniGridOneHotWrapper
            for i in range(args.procs):
                envs.append( MiniGridOneHotWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), 
                                                   seed=args.seed, **args.wrapper_args))
        elif (args.wrapper =='ssp-xy'):
            from wrappers import SSPMiniGridXYWrapper
            for i in range(args.procs): #***
                # envs.append( SSPMiniActionWrapper(SSPMiniGridXYWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed,
                #                      shape_out = args.ssp_dim,  length_scale=args.ssp_h, decoder_method = 'from-set'), 
                #                                   seed=args.seed, shape_out=args.ssp_dim) )
                envs.append( SSPMiniGridXYWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args),
                                                  seed=args.seed, **args.wrapper_args,
                                     shape_out = args.ssp_dim,  length_scale=args.ssp_h, decoder_method = 'from-set') )
        elif(args.wrapper =='ssp-view'):
            from wrappers import SSPMiniGridViewWrapper
            for i in range(args.procs):
                envs.append( SSPMiniGridViewWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args),
                                                    seed=args.seed+ 10000 * i, **args.wrapper_args,
                                     shape_out = args.ssp_dim, length_scale=args.ssp_h, decoder_method = 'from-set') )    
        elif(args.wrapper =='ssp-lang'):
             from wrappers import SSPBabyAIViewWrapper
             for i in range(args.procs):
                 envs.append( SSPBabyAIViewWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args),
                                                   seed=args.seed, shape_out = args.ssp_dim, 
                                                   length_scale=args.ssp_h, decoder_method = 'from-set', **args.wrapper_args,) )  
        else:
            exec(f"from minigrid.wrappers import {args.wrapper} as wrapper")
            for i in range(args.procs):
                envs.append(wrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args)))
    elif 'MiniWorld' in args.env: 
        if (args.wrapper =='ssp-xy'):
            from wrappers import SSPMiniWorldXYWrapper
            for i in range(args.procs):
                envs.append( SSPMiniWorldXYWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed,
                                     shape_out = args.ssp_dim,  length_scale=args.ssp_h, decoder_method = 'from-set'))
        elif (args.wrapper =='xy'):
            from wrappers import MiniWorldXYWrapper
            for i in range(args.procs):
                envs.append( MiniWorldXYWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed))  
        elif (args.wrapper =='one-hot'):
            from wrappers import MiniWorldOneHotWrapper
            for i in range(args.procs):
                envs.append( MiniWorldOneHotWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args), seed=args.seed))  
        else:
            exec(f"from miniworld.wrappers import {args.wrapper} as wrapper")
            for i in range(args.procs):
                envs.append(wrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args)))
    elif ('maze' in args.env):
        if args.wrapper=='one-hot':
            from wrappers import MazeOneHotWrapper
            for i in range(args.procs):
                envs.append(MazeOneHotWrapper(utils.make_env(args.env, args.seed + 10000 * i, **args.env_args)))
    
    txt_logger.info("Environments loaded\n")
    
    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"args": args, "num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")
    
    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    
    if args.input =='auto':
        if len(obs_space['image']) ==1: ##check
            args.input = 'flat'
        else:
            args.input = 'image'
    
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")
    
    # Load model
    is_sr = (args.algo == "sr") or (args.algo == "sr-ppo")
    if is_sr:
        model = SRModel(obs_space, envs[0].action_space, args.mem, args.text, args.normalize_embeddings,
                        args.input, args.feature_learn, 
                        obs_space_sampler=envs[0].observation_space,
                        critic_hidden_size=args.critic_hidden_size,
                        actor_hidden_size=args.actor_hidden_size, 
                        feature_hidden_size=args.feature_hidden_size,
                        feature_size=args.feature_size, feature_learn_hidden_size=args.feature_learn_hidden_size,
                        ssp_dim=args.ssp_dim, ssp_h=args.ssp_h)
    else:
        model = ACModel(obs_space, envs[0].action_space, args.mem, args.text,args.normalize_embeddings,
                        args.input, obs_space_sampler=envs[0].observation_space,
                        critic_hidden_size=args.critic_hidden_size,
                        actor_hidden_size=args.actor_hidden_size, 
                        feature_hidden_size=args.feature_hidden_size,
                        feature_size=args.feature_size,
                        ssp_dim=args.ssp_dim, ssp_h=args.ssp_h)
    if "model_state" in status:
        model.load_state_dict(status["model_state"])

    model.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(model))
    
    # Load algo
    #reshape_reward = lambda o,a,r,d: -0.1 if r==0 else 10
    if args.algo == "a2c":
        algo = A2CAlgo(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef,args.entropy_decay, args.value_loss_coef, args.dissim_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = PPOAlgo(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef,args.entropy_decay, args.value_loss_coef,args.dissim_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    elif args.algo == "sr":
        from algos.sr_a2c import SRAlgo
        algo = SRAlgo(envs, model, args.feature_learn, device, args.frames_per_proc, args.discount, args.lr_a,args.lr_f,args.lr_sr,args.lr_r, args.gae_lambda,
                                args.dissim_coef, args.entropy_coef,args.entropy_decay, 
                                args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, args.memory_cap, args.batch_size, preprocess_obss)
    
    elif args.algo == "sr-ppo":
        from algos.sr_ppo import SRPPOAlgo
        algo = SRPPOAlgo(envs, model, args.feature_learn, device, args.frames_per_proc, args.discount, args.lr_a,args.lr_f,args.lr_sr,args.lr_r, args.gae_lambda,
                                args.dissim_coef, args.entropy_coef,args.entropy_decay, 
                                args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, args.memory_cap, args.epochs, args.batch_size, args.clip_eps , preprocess_obss,None)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))
    
    if is_sr:
        if ("sr_optimizer_state" in status) and args.load_optimizer_state:
            algo.sr_optimizer.load_state_dict(status["sr_optimizer_state"])
        if ("reward_optimizer_state" in status) and args.load_optimizer_state:
            algo.reward_optimizer.load_state_dict(status["reward_optimizer_state"])
        if ("actor_optimizer_state" in status) and args.load_optimizer_state:
            algo.actor_optimizer.load_state_dict(status["actor_optimizer_state"])
        if ("feature_optimizer_state" in status) and args.load_optimizer_state:
            algo.feature_optimizer.load_state_dict(status["feature_optimizer_state"])
    else:
        if ("optimizer_state" in status) and args.load_optimizer_state:
            algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")
    
    
    if args.wandb:
        from collections import deque
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            ) from e

        run_name = f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"
        tags = [*args.wandb_tags, "successor-features-A2C"]
        wandbrun = wandb.init(
            name=run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            tags=tags,
            config=vars(args),
            sync_tensorboard=True,  # auto-upload tensorboard metrics
            monitor_gym=False,  # do not auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        ep_info_buffer = deque(maxlen=100)

    
    
    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    first_line=num_frames==0
    
    
    # keybaordstop = False

    # def handler(sig, frame):
    #     global keybaordstop
    #     keybaordstop = True
    
    # signal.signal(signal.SIGINT, handler)
    
    while (num_frames < args.frames):# and not keybaordstop:
        

        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()
    
        num_frames += logs["num_frames"]
        update += 1
        
        if args.wandb:
            [ep_info_buffer.append(l) for l in logs["return_per_episode"]];
            wandb.log({"global_step": num_frames, "rollout/ep_rew_mean": utils.safe_mean(ep_info_buffer)})
    
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
        
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | pL {:.3f} | srL {:.3f} | fL {:.3f} |  rL {:.3f} | ∇ {:.3f}"
                    .format(*data))
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
            if is_sr and args.feature_learn!='combined':
                status = {"num_frames": num_frames, "update": update,
                      "model_state": model.state_dict(),
                      "reward_optimizer_state": algo.reward_optimizer.state_dict(),
                      "sr_optimizer_state": algo.sr_optimizer.state_dict(), "actor_optimizer_state": algo.actor_optimizer.state_dict()}
                if (args.feature_learn!="none"):
                      status["feature_optimizer_state"] = algo.feature_optimizer.state_dict()
            else:
                status = {"num_frames": num_frames, "update": update,
                      "model_state": model.state_dict(), 
                      "optimizer_state": algo.optimizer.state_dict()}
                
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
       

            
    if args.plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        data = pd.read_csv(model_dir + "/log.csv")
        # data['avg_return'] = data.return_mean.copy().rolling(100).mean()
        sns.lineplot(x="frames", y='return_mean', label="Return", data=data)
        # sns.lineplot(x="frames",y="avg_return",
        #               label="Moving Avg",data=data)
        plt.xlabel("Frames observed", size=14)
        plt.ylabel("Average Return", size=14)
        plt.title( f"{args.env}, {args.algo}")
    
    if False: #args.n_test_episodes >0: # need to fix agent to use all new model init inputs
        env=envs[0]
        agent = utils.Agent(obs_space, env.action_space, model_dir, model_name =args.algo,
                        device=device, argmax=True, num_envs=1, use_memory=args.mem, use_text=args.text,
                        input_type = args.input, feature_learn = args.feature_learn, preprocess_obss=preprocess_obss)
        test_episode_returns = np.zeros(args.n_test_episodes)
        for episode in range(args.n_test_episodes):
            obs,_ = env.reset()
            n_steps = 0
            while n_steps<500:
                n_steps += 1
                action = agent.get_action(obs)
                obs, reward, t1,t2, _ = env.step(action)
                test_episode_returns[episode] += reward
                done = t1 or t2
                agent.analyze_feedback(reward, done)
        
                if done:   
                    break
        txt_logger.info("Average test return: " + str(np.mean(test_episode_returns)) 
                        + " (" + str(np.min(test_episode_returns)) + ", " + str(np.max(test_episode_returns)) + ")")
    else:
        try:# fix for case where modle is loaed
            test_episode_returns = data[4]
        except:
            test_episode_returns = None
    
    if args.wandb:
        wandbrun.finish()
    csv_file.close()
    tb_writer.close()
        
    return model_name, test_episode_returns


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--conf', type=str, default='default_config.yml',
        help='Configuration file'
    )
    args, remaining = parser.parse_known_args()
    defaults = {}
    with open(args.conf, 'r') as f: # load the defaults
        defaults = yaml.load(f, Loader=yaml.FullLoader)
        
    # Parameters to set-up env, algo type
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo | sr | sr-ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--input", type=str, default="auto",
                        help="format of input:  auto | image | flat | ssp | none (default: auto)")
    parser.add_argument("--wrapper", type=str, default="none",
                        help="format of input:  none | ssp-xy | ssp-auto | one-hot | FullyObsWrapper | RGBImgObsWrapper | OneHotPartialObsWrapper | DirectionObsWrapper (default: non)")
    parser.add_argument("--env-args", type=json.loads, default={'render_mode': 'rgb_array'},
                        help="")
    parser.add_argument("--wrapper-args", type=json.loads, default={},
                        help="")

    # General algo parameters
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--procs", type=int, default=5,
                        help="number of processes (default: 5)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO & reward function learning in SR (default: 256)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for all (default: 0.001)")
    
    # Logging parameters
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--n_test_episodes", type=int, default=10,
                        help="number eps during test phase (default: 10)")
    parser.add_argument("--load-optimizer-state", type=bool, default=False,
                        help="If True and a logs for this model (defined by model arg) exist then load the optimizer info from last run. Otherwise do not.")
    parser.add_argument("--plot", type=bool, default=False,
                        help="If True, plot mean return after training")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="")
    
    # For SR: can just give lr (above) or set different learning rates for the different parts
    parser.add_argument("--lr_f", type=float, default=None,
                        help="learning rate for feature (default: 0.001)")
    parser.add_argument("--lr_a", type=float, default=None,
                        help="learning rate for actor (default: 0.001)")
    parser.add_argument("--lr_sr", type=float, default=None,
                        help="learning rate for SR (default: 0.001)")
    parser.add_argument("--lr_r", type=float, default=None,
                        help="learning rate for reward (default: 0.00001)")
    parser.add_argument("--normalize-embeddings", type=bool, default=False,
                        help="whether or not to normlize embeddings")
    
    # Model parameters/options
    parser.add_argument("--feature-learn", type=str, default="cm",
                        help="method for feature learning:  cm | icm | latent | lap | aenc | none (default: cm)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")

    # Loss computation & optimzer parameters
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.0005,
                        help="entropy term coefficient (default: 0.0005)")
    parser.add_argument("--entropy-decay", type=float, default=0.,
                        help="entropy decay coefficient (default: 0, no decay)")
    parser.add_argument("--memory-cap", type=int, default=100000,
                        help=" (default: 100000)")
    parser.add_argument("--dissim-coef", type=float, default=0.,
                        help="state dis-similarity coefficient, only use with ssp obs or env wrappers (default: 0)")
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

    # SSP parameters: if input=ssp or wrapper is an SSP type
    parser.add_argument("--ssp-dim", type=int, default=151,
                        help="Dim of spp (default: 151)")
    parser.add_argument("--ssp-h", type=float, default=None,
                        help="Length scale of spp representation (default: None, it auto-selects")


    # Arch parameters
    parser.add_argument("--critic-hidden-size", type=int, default=64,
                        help="# of neurons in hidden layer of either the critic network or the SR network (default: 64)")
    parser.add_argument("--actor-hidden-size", type=int, default=64,
                        help="# of neurons in hidden layer of the actor network (default: 64)")
    parser.add_argument("--feature-hidden-size", type=int, default=64,
                        help="# of neurons in hidden layer of the feature network, only applicable with input-type = flat (default: 256)")
    parser.add_argument("--feature-size", type=int, default=64,
                        help="size of features, only applicable with input-type = flat (default: 64)")
    parser.add_argument("--feature-learn-hidden-size", type=int, default=64,
                        help="# of neurons in hidden layer of the feature learner network, only applicable with sr algo  (default: 64)")

    # wandb 
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument("--wandb-project-name", type=str, default="sfa2c", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument(
        "-tags", "--wandb-tags", type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: -tags optimized pr-123"
    )
    
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    
    # with open('default_config.json', 'w') as f:
    #     json.dump(vars(args), fe)
    # with open('default_config.yml', 'w') as outfile:
    #     yaml.dump(vars(args), outfile, default_flow_style=False)
    
    run(args)

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