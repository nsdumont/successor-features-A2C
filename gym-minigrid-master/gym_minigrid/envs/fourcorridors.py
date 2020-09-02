#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class FourCorridorsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, n=6, goal_corridor=None):
        self.n = n
        self._agent_default_pos = (1+n,1+n)
        if goal_corridor==1:
            goal_pos = (1, 1+n)
        elif goal_corridor==2:
            goal_pos = (1+n, 2*n+1)
        if goal_corridor==3:
            goal_pos = (2*n+1, 1+n)
        if goal_corridor==4:
            goal_pos = (1+n, 1)
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=2*n+1+2, max_steps=100)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        rect_w = width // 2
        n = self.n
        
        self.grid.wall_rect(1,1,n,n)
        self.grid.wall_rect(1,2+n,n,n)
        self.grid.wall_rect(2+n,1,n,n)
        self.grid.wall_rect(2+n,2+n,n,n)



        # Randomize the player start position and orientation
        self.agent_pos = self._agent_default_pos
        self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
    

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info

class FourCorridorsEnv1(FourCorridorsEnv):
    def __init__(self, **kwargs):
        super().__init__(goal_corridor=1, **kwargs)
        
class FourCorridorsEnv2(FourCorridorsEnv):
    def __init__(self, **kwargs):
        super().__init__(goal_corridor=2, **kwargs)
        
class FourCorridorsEnv3(FourCorridorsEnv):
    def __init__(self, **kwargs):
        super().__init__(goal_corridor=3, **kwargs)
        
class FourCorridorsEnv4(FourCorridorsEnv):
    def __init__(self, **kwargs):
        super().__init__(goal_corridor=4, **kwargs)

register(
    id='MiniGrid-FourCorridors1-v0',
    entry_point='gym_minigrid.envs:FourCorridorsEnv1'
)

register(
    id='MiniGrid-FourCorridors2-v0',
    entry_point='gym_minigrid.envs:FourCorridorsEnv2'
)


register(
    id='MiniGrid-FourCorridors3-v0',
    entry_point='gym_minigrid.envs:FourCorridorsEnv3'
)


register(
    id='MiniGrid-FourCorridors4-v0',
    entry_point='gym_minigrid.envs:FourCorridorsEnv4'
)



# import gym
# import gym_minigrid
# import matplotlib.pyplot as plt
# env = gym.make('MiniGrid-FourCorridors1-v0')
# plt.imshow(env.render('human'))

# state, reward, done,_ =env.step(2)
# plt.imshow(env.render('human'))