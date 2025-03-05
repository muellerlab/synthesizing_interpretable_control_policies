"""Finds a policy for the pendulum swingup task.

On every iteration, improve policy_v1 over the policy_vX methods from previous iterations.
Make only small changes. Try to make the code short. 
"""

import numpy as np
import funsearch
import re
from dm_control import suite

METHOD_MATCHER = re.compile(r"def policy_v\d\(.*?\) -> float:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
METHOD_NAME_MATCHER = re.compile(r"policy_v\d+")
method_str = "def policy_v"

@funsearch.run
def solve(num_runs) -> float:
  """Returns the reward for a policy.
  """
  env = suite.load(domain_name="pendulum", task_name="swingup")
  obs_spec = env.observation_spec()
  action_spec = env.action_spec()
  avg_reward = 0.0
  for _ in range(num_runs):
    time_step = env.reset()
    initialize_to_zero(env)
    total_reward = 0.0
    obs = concatenate_obs(time_step, obs_spec)
    for _ in range(1000):
      cos_theta = time_step.observation['orientation'][0]
      sin_theta = -time_step.observation['orientation'][1]
      theta = np.arctan2(sin_theta, cos_theta)
      action = policy(obs)
      action = np.clip(action, -1, 1)
      time_step = env.step(action)
      total_reward += 1.0 - np.abs(theta)/np.pi - 0.1*np.abs(action) #- 0.1*np.abs(obs[2])
      if np.abs(theta) < 0.5:
        total_reward += 1.0
      obs = concatenate_obs(time_step, obs_spec)
    avg_reward += total_reward
  return avg_reward / num_runs

def concatenate_obs(time_step, obs_spec):
  return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

def initialize_to_zero(env):
  env.physics.named.data.qpos['hinge'][0] = np.pi
  env.physics.named.data.qvel['hinge'][0] = 0.0
  env.physics.named.data.qacc['hinge'][0] = 0.0
  env.physics.named.data.qacc_smooth['hinge'][0] = 0.0
  env.physics.named.data.qacc_warmstart['hinge'][0] = 0.0
  env.physics.named.data.actuator_moment['torque'][0] = 0.0
  env.physics.named.data.qfrc_bias['hinge'][0] = 0.0

@funsearch.evolve
def policy(obs: np.ndarray) -> float:
  """Returns an action between -1 and 1.
  obs size is 3.
  """

  x1 = np.arctan2(-obs[1], obs[0])
  x2 = obs[2]
  action = 0.0
  return action