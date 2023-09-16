import types
import gymnasium as gym
# from gymnasium import error, spaces, utils
# from gymnasium.spaces import Space
from gymnasium.spaces import Box, Discrete

import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPBox, SSPDiscrete, SSPSequence, SSPDict
    

class SSPEnvWrapper(gym.Wrapper):
    """ Env wrapper for using SSP observation and/or action_spaces

    """

    def __init__(
        self,
        env: gym.Env,
        ssp_obs_space = None,
        ssp_action_space = None,
        auto_convert_obs_space = False,
        auto_convert_action_space = False,
        shape_out = None,
        **kwargs
    ):
        """
        Args:
            env (Env): The environment to apply the preprocessing
            ssp_obs_space (SSPBox | SSPDiscrete | SSPSequence | SSPDict | None): 
                    If None use default (non-SSP) obs space
            ssp_action_space (SSPBox | SSPDiscrete | SSPSequence | SSPDict | None): 
                    If None use default (non-SSP) action space
            auto_convert_spaces (bool, Default is False): If True then SSP spaces
                    will be generated automatically from the env's obs & action 
                    spaces. If True requries that shape_out is supplied.
                    Does not work for Dict or Sequence spaces.
        
        Note: If using SSPDict for either, you must define the encoding and 
             decoding methods using ssp_dict.set_encode & ssp_dict.set_decode
             and also set_mep_to_dict and set_map_from_dict if env does not have 
             a Dict obs/action space by default
        """

        gym.Wrapper.__init__(self, env)
        
        if shape_out is not None:
            assert(type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"
            
        # Set-up observation space
        if ssp_obs_space is not None:
            self.observation_space = ssp_obs_space
        elif auto_convert_obs_space:
            if type(env.observation_space) == Box:
                self.observation_space = SSPBox(
                        low = env.observation_space.low,
                        high = env.observation_space.high,
                        shape_in = env.observation_space.shape,
                        shape_out = shape_out,
                        dtype =env.observation_space.dtype,#seed = env.observation_space.seed()[0],
                        **kwargs)
            elif type(env.observation_space) == Discrete:
                self.observation_space = SSPDiscrete(
                        n = env.observation_space.n,
                        shape_out = shape_out,
                        dtype =env.observation_space.dtype,#seed = env.observation_space.seed()[0],
                        start = env.observation_space.start)
            else:
                NotImplementedError(f'Unsupported observation space for automatic SSP conversion: {env.observation_space}')
        else:
            self.observation_space = env.observation_space
            def passthrough(self, x):
                return x
            self.observation_space.encode = types.MethodType( passthrough, self.observation_space )
            self.observation_space.decode = types.MethodType( passthrough, self.observation_space )
            
        # Set-up action space
        if ssp_action_space is not None:
            self.action_space = ssp_action_space
        elif auto_convert_action_space:
            if type(env.action_space) == Box:
                self.action_space = SSPBox(
                        low = env.action_space.low,
                        high = env.action_space.high,
                        shape_in = env.action_space.shape,
                        shape_out = shape_out,
                        dtype =env.action_space.dtype,#seed = env.action_space.seed()[0],
                        **kwargs)
            elif type(env.action_space) == Discrete:
                self.action_space = SSPDiscrete(
                        n = env.action_space.n,
                        shape_out = shape_out,
                        dtype =env.action_space.dtype,#seed = env.action_space.seed()[0],
                        start = env.action_space.start)
            else:
                NotImplementedError(f'Unsupported observation space for automatic SSP conversion: {env.observation_space}')
        else:
            self.action_space = env.action_space
            def passthrough(self, x):
                return x
            self.action_space.encode = types.MethodType( passthrough, self.action_space )
            self.action_space.decode = types.MethodType( passthrough, self.action_space )
            
        

    def step(self, ssp_action):
        """Applies the preprocessing for an :meth:`env.step`."""
        # Convert input action from ssp space to orignal env space
        action = self.action_space.decode(ssp_action).reshape(self.env.action_space.shape)
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Record the decoded action taken & non-ssp obs
        info["action"] = action 
        info["obs"] = obs 
        # Convert output obs from orignal space to ssp space
        ssp_obs = self.observation_space.encode(obs).reshape(-1)
        return ssp_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment"""
        # NoopReset
        obs, reset_info = self.env.reset(**kwargs)
        ssp_obs = self.observation_space.encode(obs).reshape(-1)
        return ssp_obs, reset_info


