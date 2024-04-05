import numpy as np
import gymnasium as gym
import gym_maze


def return_maze(maze):

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
            
    return all_cells
                    

envs = np.array([ 'maze-sample-5x5-v0', 'maze-sample-6x6-v0', 'maze-sample-7x7-v0',
        'maze-sample-8x8-v0', 'maze-sample-9x9-v0', 'maze-sample-10x10-v0',
        'maze-sample-11x11-v0', 'maze-sample-12x12-v0','maze-sample-15x15-v0',
        'maze-sample-20x20-v0' ])

    
optim_steps = []
optim_rewards = []
for i, env in enumerate(envs):
    gymenv = gym.make(env)
    gymenv.reset()
    maze = return_maze(gymenv.maze_view.maze)
    
    max_steps = 1000
    start = (0,0)
    end = (gymenv.maze_view.maze.MAZE_W-1,gymenv.maze_view.maze.MAZE_H-1)
    explored = []
    n_steps = []
    def solve(pt, i):
        explored.append(pt)
        if (pt==end) or (i>= max_steps):
            n_steps.append(i)
            return pt
        cell = maze[pt[0]][pt[1]]
        next_cells = [[], [], [], []]
        if (cell['N']==0) and (pt[1]>0):
            next_pt = (pt[0], pt[1]-1)
            if next_pt not in explored:
                next_cells[0] = solve(next_pt, i+1)
        if (cell['S']==0) and (pt[1]<gymenv.maze_view.maze.MAZE_H-1):
            next_pt = (pt[0], pt[1]+1)
            if next_pt not in explored:
                next_cells[1] = solve(next_pt, i+1)
        if (cell['E']==0) and (pt[0]<gymenv.maze_view.maze.MAZE_W-1):
            next_pt = (pt[0]+1, pt[1])
            if next_pt not in explored:
                next_cells[2] = solve(next_pt, i+1)
        if (cell['W']==0) and (pt[0]>0):
            next_pt = (pt[0]-1, pt[1])
            if next_pt not in explored:
                next_cells[3] = solve(next_pt, i+1)
        return [pt, next_cells]
    
    soln = solve(start,0)
    optim_steps.append(np.min(n_steps))
    optim_rewards.append(1 - 0.9*(optim_steps[-1]/gymenv._max_episode_steps))