from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyNoGoalEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
       # self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        #self.mission = "get to the green goal square"

class EmptyNoGoalEnv5x5(EmptyNoGoalEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)

class EmptyNoGoalRandomEnv5x5(EmptyNoGoalEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyNoGoalEnv6x6(EmptyNoGoalEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)

class EmptyNoGoalRandomEnv6x6(EmptyNoGoalEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyNoGoalEnv16x16(EmptyNoGoalEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)

register(
    id='MiniGrid-EmptyNoGoal-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyNoGoalEnv5x5'
)

register(
    id='MiniGrid-EmptyNoGoal-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyNoGoalRandomEnv5x5'
)

register(
    id='MiniGrid-EmptyNoGoal-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyNoGoalEnv6x6'
)

register(
    id='MiniGrid-EmptyNoGoal-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyNoGoalRandomEnv6x6'
)

register(
    id='MiniGrid-EmptyNoGoal-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyNoGoalEnv'
)

register(
    id='MiniGrid-EmptyNoGoal-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyNoGoalEnv16x16'
)
