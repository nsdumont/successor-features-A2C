import numpy as np
import matplotlib.pyplot as plt
import sys,os
os.chdir("..")
import utils
from models.model import ACModel
import torch
import gymnasium as gym
import gym_maze
from wrappers import SSPEnvWrapper
import colorsys
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as mcolors



def get_value_policy(plot_env, n_pts=50):
    plot_model = plot_env + '_ppo_ssp_0' 
    maze_size = float(plot_env.split('-')[2].split('x')[0])
    plot_model_dir = utils.get_model_dir(plot_model)
    status = utils.get_status(plot_model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    # in future, can use status['args'] with new train.py, for old ones:
    if "args" in status:
        env= SSPEnvWrapper(utils.make_env(plot_env, 0), seed=0,
                                auto_convert_obs_space = True, auto_convert_action_space=False,
                                shape_out = status['args'].ssp_dim, length_scale=status['args'].ssp_h,
                                decoder_method = 'from-set')
        obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
        model = ACModel(obs_space, env.action_space, status['args'].mem, status['args'].text,
                        status['args'].normalize_embeddings,
                        status['args'].input, obs_space_sampler=env.observation_space,
                        critic_hidden_size=status['args'].critic_hidden_size,
                        actor_hidden_size=status['args'].actor_hidden_size, 
                        feature_hidden_size=status['args'].feature_hidden_size,
                        feature_size=status['args'].feature_size)
    else:
        actor_hidden_size= 64
        critic_hidden_size= 128
        feature_hidden_size= 64
        feature_size=64
        ssp_dim= 55
        ssp_h= 0.18790019414748999*(maze_size/5)**2
        
        env= SSPEnvWrapper(utils.make_env(plot_env, 0), seed=0,
                                auto_convert_obs_space = True, auto_convert_action_space=False,
                                shape_out = ssp_dim, length_scale=ssp_h,
                                decoder_method = 'from-set')
        obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
        model = ACModel(obs_space, env.action_space, False, False, False,
                            "flat", obs_space_sampler=env.observation_space,
                            critic_hidden_size=critic_hidden_size,
                            actor_hidden_size=actor_hidden_size, 
                            feature_hidden_size=feature_hidden_size,
                            feature_size=feature_size)
    print("Environment loaded\n")
    model.load_state_dict(status["model_state"])
    model.to(device)
    print("Model loaded\n")
     
    if n_pts is None:
        X,Y=np.meshgrid(np.arange(env.maze_view.maze.MAZE_W), np.arange(env.maze_view.maze.MAZE_H))
    else:
        X,Y=np.meshgrid(np.linspace(0,env.maze_view.maze.MAZE_W,n_pts), np.linspace(0,env.maze_view.maze.MAZE_H,n_pts))
    mazepts  = np.vstack([X.flatten(), Y.flatten()]).T
    ssppts = env.observation_space.encode(mazepts)
    
    preprocessed_obs = preprocess_obss(torch.tensor(ssppts), device=device)
    with torch.no_grad():
        dist, value, _ = model(preprocessed_obs, None)
    
    return env.maze_view.maze, X, Y, value.cpu().numpy(), dist.probs.cpu().numpy()
    
    

def draw_maze(maze, ax, c='k', lw=5, offset=False):

    # Scaling factors mapping maze coordinates to image coordinates

    all_cells = [[{'N': 1, 'E': 1, 'S': 1, 'W': 1} for _ in range(maze.MAZE_H) ] for _ in range(maze.MAZE_W)]
    for x in range(maze.MAZE_W):
        for y in range(maze.MAZE_H):
            cell = maze.get_walls_status(maze.maze_cells[x,y])
            if cell['N']:
                all_cells[x][y]['N'] = 0
                if y>0:
                    all_cells[x][y-1]['S'] = 0
            if cell['S']:
                all_cells[x][y]['S'] = 0
                if y<maze.MAZE_H-1:
                    all_cells[x][y+1]['N'] = 0
            if cell['E']:
                all_cells[x][y]['E'] = 0
                if x<maze.MAZE_W-1:
                    all_cells[x+1][y]['W'] = 0
            if cell['W']:
                all_cells[x][y]['W'] = 0
                if x>0:
                    all_cells[x-1][y]['E'] = 0
            
    if offset:
        shift = 0.5
    else:
        shift = 0
    for x in range(maze.MAZE_W):
        for y in range(maze.MAZE_H):
            walls = all_cells[x][y]
            if walls['N']:
                x1, y1, x2, y2 = x-shift, y -shift, x + 1-shift, y-shift
                ax.plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=lw)
            if walls['S']:
                x1, y1, x2, y2 = x-shift , y + 1-shift, x +1-shift, y + 1-shift  
                ax.plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=lw)
            if walls['W']:
                x1, y1, x2, y2 = x-shift, y-shift , x -shift, y + 1-shift
                ax.plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=lw)
            if walls['E']:
                x1, y1, x2, y2 = x+1-shift, y-shift, x+1-shift, y+1-shift
                ax.plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=lw)
                    
       

def save_maze(maze, offset=True):

    # Scaling factors mapping maze coordinates to image coordinates

    all_cells = [[{'N': 1, 'E': 1, 'S': 1, 'W': 1} for _ in range(maze.MAZE_H) ] for _ in range(maze.MAZE_W)]
    for x in range(maze.MAZE_W):
        for y in range(maze.MAZE_H):
            cell = maze.get_walls_status(maze.maze_cells[x,y])
            if cell['N']:
                all_cells[x][y]['N'] = 0
                if y>0:
                    all_cells[x][y-1]['S'] = 0
            if cell['S']:
                all_cells[x][y]['S'] = 0
                if y<maze.MAZE_H-1:
                    all_cells[x][y+1]['N'] = 0
            if cell['E']:
                all_cells[x][y]['E'] = 0
                if x<maze.MAZE_W-1:
                    all_cells[x+1][y]['W'] = 0
            if cell['W']:
                all_cells[x][y]['W'] = 0
                if x>0:
                    all_cells[x-1][y]['E'] = 0
            
    shift = 0
    all_walls = []
    for x in range(maze.MAZE_W):
        for y in range(maze.MAZE_H):
            walls = all_cells[x][y]
            if walls['N']:
                x1, y1, x2, y2 = x-shift, y -shift, x + 1-shift, y-shift
                all_walls.append([[x1, x2], [y1, y2]])
            if walls['S']:
                x1, y1, x2, y2 = x-shift , y + 1-shift, x +1-shift, y + 1-shift  
                all_walls.append([[x1, x2], [y1, y2]])
            if walls['W']:
                x1, y1, x2, y2 = x-shift, y-shift , x -shift, y + 1-shift
                all_walls.append([[x1, x2], [y1, y2]])
            if walls['E']:
                x1, y1, x2, y2 = x+1-shift, y-shift, x+1-shift, y+1-shift
                all_walls.append([[x1, x2], [y1, y2]])
    dir = '/home/ns2dumon/Documents/Github/gym-continuous-maze/gym_continuous_maze/maze_samples/'
    np.save(dir + f'maze2d_{maze.MAZE_W}x{maze.MAZE_H}.npy', np.array(all_walls), allow_pickle=False, fix_imports=True)
                    
       
                                     
        # ax.plot(0, 0, 0, maze.MAZE_H, color=c, linewidth=lw)
        # ax.plot(0, 0, maze.MAZE_W, 0, color=c, linewidth=lw)
       



def plot_value(maze,X,Y,value,ax,bar=False,offset=False,shading='gouraud',lw=2):
    cs=ax.pcolormesh(X,Y,value.reshape(X.shape),cmap= 'Greens', shading=shading)
    if bar:
        ax_inset = ax.inset_axes([0.7,0.3,0.1,0.4])
        cb=plt.colorbar(cs,cax=ax_inset,drawedges=True)
        cb.outline.set_linewidth(lw)
        cb.dividers.set_linewidth(0)
    draw_maze(maze, ax, c='k', lw=lw,offset=offset)
    
    
    
    # ax.set_xlim([-1,maze.MAZE_W+1])
    # ax.set_ylim([maze.MAZE_H+1,-1])
    
def plot_policy(maze,X,Y,probs,ax, wheel=True,offset=False,shading='gouraud',lw=2):
    directions = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    vecs = probs @ directions
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    magnitudes = np.sqrt(np.sum(vecs**2, axis=1)) 
    hue = (angles + np.pi) / (2 * np.pi)  # Normalize between 0 and 1
    saturation = magnitudes / np.max(magnitudes)
    values = np.ones_like(hue)
    hsv_array = np.stack((hue, saturation, values), axis=1)
    rgb_array = mcolors.hsv_to_rgb(hsv_array)
    rgb_matrix = rgb_array.reshape((X.shape[0],X.shape[1], 3))
    
    ax.pcolormesh(X, Y, rgb_matrix, shading=shading)
    draw_maze(maze, ax, c='k', lw=lw,offset=offset)

    if wheel:
        # Add a polar inset
        ax_inset = ax.inset_axes([0.7,0.3,0.3,0.3], projection='polar')
        ax_inset.set_theta_direction(-1)   # Change the direction to clockwise
        ax_inset.set_theta_offset(-np.pi / 2)
        # Create a color wheel
        num_colors = 100
        theta = np.linspace(0, 2*np.pi, num_colors)
        rgb_wheel = mcolors.hsv_to_rgb(np.column_stack((theta/(2*np.pi), np.ones(num_colors), np.ones(num_colors))))
        for sat in np.linspace(0, 1, 10):  # 10 saturation levels from center to edge
            rgb_wheel = mcolors.hsv_to_rgb(np.column_stack((theta/(2*np.pi), sat * np.ones(num_colors), np.ones(num_colors))))
            ax_inset.bar(theta, np.ones(num_colors) * sat, color=rgb_wheel, width=2*np.pi/num_colors, bottom=1-sat)
        # ax_inset.set_xticklabels([])
        # ax_inset.set_yticklabels([])
        # ax_inset.spines['polar'].set_visible(False)
        # ax_inset.bar(theta, np.ones(num_colors), color=rgb_wheel, width=2*np.pi/num_colors)
        # ax_inset.set_axis_off()
        # Remove x and y ticks
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        
        # Set the visibility of the polar frame (the bold black outline)
        ax_inset.spines['polar'].set_visible(True)
        ax_inset.spines['polar'].set_linewidth(lw)


envs = np.array([ 'maze-sample-5x5-v0', 'maze-sample-6x6-v0', 'maze-sample-7x7-v0',
        'maze-sample-8x8-v0', 'maze-sample-9x9-v0', 'maze-sample-10x10-v0',
        'maze-sample-11x11-v0', 'maze-sample-12x12-v0','maze-sample-15x15-v0',
        'maze-sample-20x20-v0' ])
for env in envs:
    gymenv = gym.make(env)
    gymenv.reset()
    maze = gymenv.maze_view.maze
    save_maze(gymenv.maze_view.maze);
    
n_pts = None
include_value = True
plot_envs =  ['maze-sample-5x5-v0','maze-sample-8x8-v0','maze-sample-12x12-v0']
save = False
lw = 1
if n_pts is None:
    offset=True
    shading='auto'
else:
    offset=False
    shading='gouraud'

if include_value:
    fig,axs = plt.subplots(2,3, figsize=(6.5,4),
                           gridspec_kw={'width_ratios': [1,1,1.4], 'wspace': 0.01, 'hspace':0.01})
    for i,plot_env in enumerate(plot_envs):
        maze, X, Y, values, probs = get_value_policy(plot_env,n_pts=n_pts)
        if i<2:
            plot_value(maze,X,Y,values,axs[0,i],bar=False,offset=offset,shading=shading,lw=lw)
            plot_policy(maze,X,Y,probs,axs[1,i],wheel=False,offset=offset,shading=shading,lw=lw)
            for j in  range(2):
                axs[j,i].set_xlim([-maze.MAZE_W/5,maze.MAZE_W*(1 +1/5)])
                axs[j,i].set_ylim([maze.MAZE_H*(1+1/5),-maze.MAZE_H/5])
        else:
            plot_value(maze,X,Y,values,axs[0,i],bar=True,offset=offset,shading=shading,lw=lw)
            plot_policy(maze,X,Y,probs,axs[1,i],wheel=True,offset=offset,shading=shading,lw=lw)
            for j in range(2):
                axs[j,i].set_xlim([-maze.MAZE_W/5,maze.MAZE_W*(1 +3/5)])
                axs[j,i].set_ylim([maze.MAZE_H*(1+1/5),-maze.MAZE_H/5])
        axs[0,i].set_axis_off()
        axs[1,i].set_axis_off()
    fig.tight_layout()
    fig.text(0.15,0.9,"\\textbf{A} $\quad$ Learned value function",size=12)
    fig.text(0.15,0.5,"\\textbf{B} $\quad$ Learned policices",size=12)
    if save:
        utils.save(fig,'figures/maze_values_policies.pdf')
        utils.save(fig,'figures/maze_values_policies.png')
else:
    fig,axs = plt.subplots(1,3, figsize=(6.5,2),
                           gridspec_kw={'width_ratios': [1,1,1.4]})
    for i,plot_env in enumerate(plot_envs):
        maze, X, Y, values, probs = get_value_policy(plot_env,n_pts=n_pts)
        if i<2:
            plot_policy(maze,X,Y,probs,axs[i],wheel=False,offset=offset,shading=shading)
            axs[i].set_xlim([-maze.MAZE_W/5,maze.MAZE_W*(1 +1/5)])
            axs[i].set_ylim([maze.MAZE_H*(1+1/5),-maze.MAZE_H/5])
        else:
            plot_policy(maze,X,Y,probs,axs[i],wheel=True,offset=offset,shading=shading)
            axs[i].set_xlim([-maze.MAZE_W/5,maze.MAZE_W*(1 +3/5)])
            axs[i].set_ylim([maze.MAZE_H*(1+1/5),-maze.MAZE_H/5])
        axs[i].set_axis_off()
    fig.tight_layout()
    if save:
        utils.save(fig,'figures/maze_policies.pdf')
        utils.save(fig,'figures/maze_policies.png')

# plot_envs =  ['maze-sample-5x5-v0','maze-sample-8x8-v0','maze-sample-12x12-v0']
# fig,axs = plt.subplots(2,3, figsize=(6.5,4),
#                        gridspec_kw={'width_ratios': [1,1,1.2]})
# for i,plot_env in enumerate(plot_envs):
#     maze, X, Y, values, probs = get_value_policy(plot_env)
#     plot_value(maze,X,Y,values,axs[0,i])
#     if i<2:
#         plot_policy(maze,X,Y,probs,axs[1,i],wheel=False)
#         for j in range(2):
#             axs[j,i].set_xlim([-1,maze.MAZE_W+1])
#             axs[j,i].set_ylim([maze.MAZE_H+1,-1])
#     else:
#         plot_policy(maze,X,Y,probs,axs[1,i],wheel=True)
#         for j in range(2):
#             axs[j,i].set_xlim([-1,maze.MAZE_W+3])
#             axs[j,i].set_ylim([maze.MAZE_H+3,-1])
#     axs[0,i].set_axis_off()
#     axs[1,i].set_axis_off()
    
    # axs[1].set_xlim([-1,env.maze_view.maze.MAZE_W+3])
    # axs[1].set_ylim([env.maze_view.maze.MAZE_H+1,-1])

# directions = np.array([[1,0],[0,1],[-1,0],[0,-1]])
# vecs = dist.probs.cpu().numpy() @ directions
# vecs = 0.5*(vecs + 1)
# hsvs = np.hstack([vecs, 0.99*np.ones((len(mazepts),1))])
# rgbs = np.array([colorsys.hsv_to_rgb(h[0],h[1],h[2]) for h in hsvs])

# axs[1].pcolormesh(X,Y,rgbs.reshape(X.shape[0],X.shape[1],3))
# draw_maze(env.maze_view.maze, axs[1], c='k', lw=2)
# axs[1].set_xlim([-1,env.maze_view.maze.MAZE_W+3])
# axs[1].set_ylim([env.maze_view.maze.MAZE_H+1,-1])

# inax = axs[1].inset_axes([0.8,0.2,0.31,0.3], projection='polar')
# inax._direction = 2*np.pi
# norm = mpl.colors.Normalize(0.0, 2*np.pi)
# quant_steps = 2056
# cb = mpl.colorbar.ColorbarBase(inax, cmap=cm.get_cmap('hsv',quant_steps),
#                                    norm=norm,
#                                    orientation='horizontal')

# # aesthetics - get rid of border and axis labels                                   
# cb.outline.set_visible(False)                                 
# inax.set_axis_off()

