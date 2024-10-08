"""Test for pendulum swingup
"""

import numpy as np
import matplotlib.pyplot as plt
import dm_control
from dm_control import suite
from dm_control import viewer

def initialize_to_zero(env):
  env.physics.named.data.qpos['hinge'][0] = np.pi
  env.physics.named.data.qvel['hinge'][0] = 0.0
  env.physics.named.data.qacc['hinge'][0] = 0.0
  env.physics.named.data.qacc_smooth['hinge'][0] = 0.0
  env.physics.named.data.qacc_warmstart['hinge'][0] = 0.0
  env.physics.named.data.actuator_moment['torque'][0] = 0.0
  env.physics.named.data.qfrc_bias['hinge'][0] = 0.0

def policy(obs: np.ndarray) -> float:
  """Returns an action between -1 and 1.
  t is a time counter. obs size is 3.
  """
  x1 = np.arctan2(-obs[1], obs[0])
  x2 = obs[2]
  if abs(x1)    <  0.5:
    action = 5*x1 - 0.9*x2
  else:
    action = np.sign(x2)

  return action

if __name__ == "__main__":
  env = suite.load(domain_name="pendulum", task_name="swingup")  

  action_spec = env.action_spec()
  action_size = action_spec.shape[0]
  steps_offset = 0
  action_seq = np.zeros((action_size, 1000-steps_offset))

  obs_spec = env.observation_spec()

  def concatenate_obs(time_step, obs_spec):
    return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])
  
  time_step = env.reset()
  initialize_to_zero(env)
  for _ in range(steps_offset):
    env.step(0.0)

  total_reward = 0.0
  obs = concatenate_obs(time_step, obs_spec)
  obs_size = obs.shape[0]
  obs_seq = np.zeros((obs_size, 1001-steps_offset))
  obs_seq[:, 0] = obs
  time = range(1000-steps_offset)
  viewer.launch(env)

  for i in time:
    cos_theta = time_step.observation['orientation'][0]
    sin_theta = -time_step.observation['orientation'][1]
    theta = np.arctan2(sin_theta, cos_theta)
    action = policy(obs)
    action = np.clip(action, -1, 1)
    action_seq[:, i] = action
    time_step = env.step(action)
    total_reward += 1.0 - np.abs(theta)/np.pi - 0.1*np.abs(action)
    if np.abs(theta) < 0.5:
      total_reward += 1.0
    obs = concatenate_obs(time_step, obs_spec)
    obs_seq[:, i+1] = obs

  print(f"Score: {total_reward}")
  LW = 2
  start = 1
  time_plot = 0.015*np.array(time)
  plt.rcParams.update({'font.size': 15})
  plt.rcParams['text.usetex'] = True

  # Create figure and axes for subplots
  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  fig, axs = plt.subplots(3, 1, figsize=(650*px, 530*px))

  # First subplot (Action)
  axs[0].plot(time_plot[start:], action_seq.squeeze()[start:], linewidth=LW)
  axs[0].set_ylabel('$u$')
  axs[0].grid(True)

  # Second subplot (Theta)
  theta_plot = np.arctan2(-obs_seq[1, start:-1], obs_seq[0, start:-1])
  theta_plot[theta_plot < 0] += 2 * np.pi
  theta_plot -= np.pi
  theta_plot *= -1
  theta_plot += np.pi

  # Horizontal lines with text above them
  axs[1].axhline(y=2 * 180, color='black', linestyle='--', linewidth=LW)
  # axs[1].text(time[-1] * 0.5, 2 * np.pi + 0.1, '2π Line', horizontalalignment='center', verticalalignment='bottom')

  axs[1].axhline(y=180, color='black', linestyle='--', linewidth=LW)
  # axs[1].text(0.5, np.pi - 0.5, 'π Line', horizontalalignment='left', verticalalignment='bottom', transform=axs[1].get_xaxis_transform())

  # Plot theta
  axs[1].plot(time_plot[start:], theta_plot*180/np.pi, label="theta", ls="-", linewidth=LW, color="green")
  axs[1].set_ylabel(r'$\theta \, \mathrm{[deg]}$')
  axs[1].grid(True)

  # Third subplot (Theta_dot)
  axs[2].plot(time_plot[start:], obs_seq[2, start:-1], linewidth=LW, color="red")
  axs[2].set_xlabel(r'$t \, \mathrm{[s]}$')
  axs[2].set_ylabel(r'$\dot{\theta} \, \mathrm{[rad/s]}$')

  # Display the plot
  plt.tight_layout()
  plt.grid(True)
  plt.show()