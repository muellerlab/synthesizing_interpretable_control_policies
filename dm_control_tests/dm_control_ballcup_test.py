"""Test for ball in cup
"""

import numpy as np
import matplotlib.pyplot as plt
import dm_control
from dm_control import suite
from dm_control import viewer

def initialize_to_zero(env):
  env.physics.named.data.qpos['ball_x'][0] = 0.0
  env.physics.named.data.qpos['ball_z'][0] = 0.0

def policy(obs: np.ndarray, output_shape: tuple) -> np.ndarray: # best one so far (upper right corner)
  """Returns two actions between -1 and 1.
  obs size is 8.
  """
  p1 = obs[0:2]
  p2 = obs[2:4]
  v1 = obs[4:6]
  v2 = obs[6:8]

  action = np.zeros((2,))
  action[1] = -0.1
  if obs[0] < 0.2:
    action[0] = 1

  if obs[3] < -0.2:
    action[1] = 1
  elif obs[4] > 0.2:
    action[1] = -1
  elif obs[5] < -0.2:
    action[1] = -1

  if obs[6] > 0.5:
    action[1] = 1
  elif obs[7] < -0.5:
    action[1] = 1

  if obs[1] < 0.2:
    action[0] = 1

  ### user-added code ###
  # if obs[3] - obs[1] > 0.1:
  #   action[1] = action[1] - 0.1

  return action

if __name__ == "__main__":
  env = suite.load(domain_name="ball_in_cup", task_name="catch")  
  obs_spec = env.observation_spec()
  action_spec = env.action_spec()
  
  action_size = action_spec.shape[0]
  steps_offset = 0
  action_seq = np.zeros((action_size, 1000-steps_offset))

  def concatenate_obs(time_step, obs_spec):
    return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

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
  
  time_step = env.reset()
  # env initialization, optional
  initialize_to_zero(env)

  # time_step = env.step(0.0)
  total_reward = 0.0
  obs = concatenate_obs(time_step, obs_spec)
  obs[3] -= 0.3
  obs_size = obs.shape[0]
  obs_seq = np.zeros((obs_size, 1001-steps_offset))
  obs_seq[:, 0] = obs
  time = range(1000-steps_offset)

  # defined to see the system in the viewer
  def policy_timestep(time_step):
    obs = concatenate_obs(time_step, obs_spec)
    obs[3] -= 0.3
    action = policy(obs, action_spec.shape)
    return action
  
  # viewer.launch(env, policy=policy_timestep)

  for i in time:
    action = policy(obs, action_spec.shape)
    action = np.array(action, dtype=np.float64)
    action = np.clip(action, action_spec.minimum[0], action_spec.maximum[0])
    action_seq[:, i] = action
    time_step = env.step(action)
    obs = concatenate_obs(time_step, obs_spec)
    obs[3] -= 0.3
    obs_seq[:, i+1] = obs
    total_reward += time_step.reward
    total_reward += custom_reward(obs)

  print(f"Score: {total_reward}")

  # Plotting
  LW = 3
  start = 1
  plt.rcParams.update({'font.size': 22})

  # Create figure and axes for subplots
  fig = plt.figure(figsize=(10, 8))
  axs1 = plt.subplot(2, 2, 1)

  # First subplot (Action)
  axs1.plot(time[start:], action_seq[0,start:], linewidth=LW, label="action x")
  axs1.plot(time[start:], action_seq[1,start:], linewidth=LW, linestyle='--', label="action y")
  axs1.set_ylabel("actions")

  axs2 = plt.subplot(2, 2, 3)
  vel_ball = np.sqrt(obs_seq[6, start:-1]**2 + obs_seq[7, start:-1]**2)
  axs2.plot(time[start:], vel_ball, linewidth=LW, color="blue")
  axs2.set_ylabel("ball velocity")
  axs2.set_xlabel("time")

  axs3 = plt.subplot(1, 2, 2)
  # Plot positions
  axs3.plot(obs_seq[0, start:], obs_seq[1, start:], label="cup", ls="-", linewidth=LW, color="green")
  axs3.plot(obs_seq[2, start:], obs_seq[3, start:], label="ball", ls="--", linewidth=LW, color="red")
  axs3.set_ylabel("positions y")
  axs3.set_xlabel("positions x")
  axs3.legend()

  # plt.tight_layout()
  plt.show()