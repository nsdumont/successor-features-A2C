import argparse
import time
import numpy
import torch
import matplotlib.pyplot as plt
import utils
import gymnasium as gym
import yaml



#python visualize_agent.py --algo sr --env MiniGrid-Empty-6x6-v0 --model MiniGrid-Empty-6x6-v0_sr_seed1_23-05-11-17-12-46 --input image 

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--algo", type=str, default='ac',
                    help="name of the type of model")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--input", type=str, default="image",
                    help="features")
parser.add_argument("--feature-learn", type=str, default="curiosity",
                    help="feature learning")
parser.add_argument("--continous-action", type=bool, default=False,
                    help=" ")
parser.add_argument("--env-args", type=yaml.load, default={},
                    help="")


args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, seed=args.seed,env_args=args.env_args)
if args.input =='ssp':
    from minigrid.wrappers import SSPWrapper
    import nengo_ssp as ssp
    X,Y,_ = ssp.HexagonalBasis(10,10)
    d = len(X.v)
    env = SSPWrapper( env,d,X,Y)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,model_name =args.algo,
                    device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text,
                    input_type = args.input, feature_learn = args.feature_learn)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []


plt.imshow(env.render('human'))

for episode in range(args.episodes):
    obs = env.reset()

    while True:
        plt.imshow(env.render('human'))
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")