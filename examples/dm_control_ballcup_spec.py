"""Finds a policy for the ball in cup task.

On every iteration, improve policy_v1 over the policy_vX methods from previous iterations.
Make only small changes. Try to make the code short. 
"""

import numpy as np
import funsearch
import re
from dm_control import suite

METHOD_MATCHER = re.compile(r"def policy_v\d\(.*?\) -> np.ndarray:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
METHOD_NAME_MATCHER = re.compile(r"policy_v\d+")
method_str = "def policy_v"


@funsearch.run
def solve(num_runs) -> float:
  """Returns the reward for a policy.
  """
  env = suite.load(domain_name="ball_in_cup", task_name="catch") 
  obs_spec = env.observation_spec()
  action_spec = env.action_spec()

  min_reward = 1e5
  initial_xpos = np.array([-0.1, 0.0, 0.1])
  for i in range(num_runs):
    # time.sleep(.002)
    time_step = env.reset()
    initialize_env(env, initial_xpos[i%3])
    total_reward = 0.0
    obs = concatenate_obs(time_step, obs_spec)
    obs[3] -= 0.3
    for _ in range(1000):
      action = policy(obs, action_spec.shape)
      # if action is None:
      #   print("action is None")
      # action = np.array(action, dtype=np.float64)
      action = np.clip(action, action_spec.minimum[0], action_spec.maximum[0])
      time_step = env.step(action)
      obs = concatenate_obs(time_step, obs_spec)
      obs[3] -= 0.3
      total_reward += time_step.reward # +1 if ball in cup, 0 otherwise
      total_reward += custom_reward(obs)
    if total_reward < min_reward:
      min_reward = total_reward
  return min_reward

def concatenate_obs(time_step, obs_spec):
  return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

def initialize_env(env, x_pos):
  env.physics.named.data.qpos['ball_x'][0] = x_pos
  env.physics.named.data.qpos['ball_z'][0] = 0.0

def custom_reward(obs: np.ndarray) -> float:
  x_cup = obs[0]
  z_cup = obs[1]
  x_ball = obs[2]
  z_ball = obs[3]
  angle = np.arctan2(x_ball - x_cup, z_ball - z_cup)
  vx_ball = obs[6]
  vz_ball = obs[7]
  v_ball = np.sqrt(vx_ball**2 + vz_ball**2)
  reward = 1 - np.abs(angle)/np.pi
  if v_ball > 4.0:
    reward -= 0.1*v_ball
  return reward

@funsearch.evolve
def policy(obs: np.ndarray, output_shape: tuple) -> np.ndarray:
  """Returns an action between -1 and 1.
  obs size is 8. return size is 2.
  """
  action = np.zeros((2,))

  return action