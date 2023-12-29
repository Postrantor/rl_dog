# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pybullet as p
import os

# from self
from logger.logger import Logger  # get logger
from utils.replay_memory import ReplayMemory
from network.network import SoftQNetwork
from network.network import ValueNetwork
from network.network import PolicyNetwork

logger = Logger().get_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SACAgent:
  """
  An agent for the first generation of Soft Actor Critic learning algorithm
  Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
  Haarnoja et al. 2018
  """

  def __init__(self, env, lr=3e-4, replay_buffer_size=1000000):
    """
    :param env: an instance of an OpenAI Gym environment that is being learned on.
    :param lr: float, the learning rate used to update the parameters
    :param replay_buffer_size: int, size of the replay buffer
    """

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", self.device)

    try:
      action_dim = env.action_space.shape[0]
    except IndexError:
      action_dim = env.action_space.n
    try:
      observation_dim = env.observation_space.shape[0]
    except IndexError:
      observation_dim = env.observation_space.n

    self.value_net = ValueNetwork(observation_dim, 256).to(self.device)
    self.target_value_net = ValueNetwork(observation_dim,
                                         256).to(self.device)

    self.soft_q1 = SoftQNetwork(observation_dim,
                                action_dim).to(self.device)
    self.soft_q2 = SoftQNetwork(observation_dim,
                                action_dim).to(self.device)

    self.policy = PolicyNetwork(observation_dim,
                                action_dim,
                                256,
                                device=self.device).to(self.device)

    self.target_value_net.load_state_dict(self.value_net.state_dict())

    self.value_criterion = nn.MSELoss()
    self.q1_criterion = nn.MSELoss()
    self.q2_criterion = nn.MSELoss()

    self.value_optim = optim.Adam(self.value_net.parameters(), lr=lr)
    self.q1_optim = optim.Adam(self.soft_q1.parameters(), lr=lr)
    self.q2_optim = optim.Adam(self.soft_q2.parameters(), lr=lr)
    self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

    self.mem_size = replay_buffer_size
    self.replay_buffer = ReplayMemory(self.mem_size)

  def update(self, batch_size, gamma=0.99, tau=1e-2):
    """
    Update the parameters of the agent
    :param batch_size: Size of sample taken from replay memory
    :param gamma: Discount factor for calculating Q loss
    :param tau: Smoothing rate for updating target functions
    :return: None
    """

    state, action, next_state, reward, done = self.replay_buffer.sample(
        batch_size)

    state = torch.FloatTensor(state).to(self.device)
    next_state = torch.FloatTensor(next_state).to(self.device)
    action = torch.FloatTensor(action).to(self.device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

    pred_q1 = self.soft_q1(state, action)
    pred_q2 = self.soft_q2(state, action)
    pred_val = self.value_net(state).mean()
    new_action, log_prob = self.policy.evaluate(next_state)

    # Train Q using Q loss:
    # J(Q_i) = 1/2 (Q_i(s_t, a_t) - Q'(s_t, a_t))^2, where,
    # Q'(s_t, a_t) = r(s_t, a_t) + gamma * target_V(s_(t+1))
    target_val = self.target_value_net(next_state)
    target_q = reward + (1 - done) * gamma * target_val
    q1_loss = self.q1_criterion(pred_q1, target_q.detach())
    q2_loss = self.q2_criterion(pred_q2, target_q.detach())
    self.q1_optim.zero_grad()
    q1_loss.backward()
    self.q1_optim.step()
    self.q2_optim.zero_grad()
    q2_loss.backward()
    self.q2_optim.step()

    # Train V with the loss function
    # J(V) = 1/2 (V(s_t) - (Q(s_t, a_t) - log policy(a_t, s_t)))^2
    pred_new_q = torch.min(self.soft_q1(state, new_action),
                           self.soft_q2(state, new_action))
    target_val_func = pred_new_q - log_prob
    val_loss = self.value_criterion(pred_val, target_val_func.detach())
    self.value_optim.zero_grad()
    val_loss.backward()
    self.value_optim.step()

    # Train Policy with loss function
    # J(policy) = mean(log policy(a_t, s_t) - Q(s_t, a_t))
    policy_loss = (log_prob - pred_new_q).mean()
    self.policy_optim.zero_grad()
    policy_loss.backward()
    self.policy_optim.step()

    # Update the target value parameters with
    # target_param = tau * param + (1 - tau) * target_param
    for target_param, param in zip(self.target_value_net.parameters(),
                                   self.value_net.parameters()):
      target_param.data.copy_(param.data * tau + target_param.data *
                              (1.0 - tau))

  def save_policy(self, path):
    """
    Saves the state dictionary of the policy network the the specified path
    :param path: The path to save the state dictionary to
    :return: None
    """
    torch.save(self.policy.state_dict(), path)


class SAC2Agent:
  """
  Agent for the second generation of the Soft Actor Critic learning algorithm presented in
  Soft Actor-Critic Algorithms and Applications, Haarnoja et al. 2018
  Differs from SACAgent by replacing the need for a value function with an entropy temperature parameter and tuning
  this while learning
  """

  def __init__(self,
               env,
               alpha=0.1,
               alr=1e-4,
               qlr=1e-4,
               policy_lr=1e-4,
               mem_size=1e6):
    """
    :param env: an instance of an OpenAI Gym environment that is being learned on.
    :param alpha: float, initial value for alpha
    :param alr: float, the learning rate used to update alpha
    :param qlr: float, the learning rate used to update the q functions
    :param policy_lr: float, the learning rate used to update the policy function
    :param mem_size: int, size of the replay buffer, 1e6 by default
    """

    try:
      action_dim = env.action_space.shape[0]
    except IndexError:
      action_dim = env.action_space.n
    try:
      observation_dim = env.observation_space.shape[0]
    except IndexError:
      observation_dim = env.observation_space.n

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # print('--- --- --- %s', device)

    self.q1 = SoftQNetwork(observation_dim, action_dim).to(self.device)
    self.q2 = SoftQNetwork(observation_dim, action_dim).to(self.device)

    self.target_q1 = SoftQNetwork(observation_dim,
                                  action_dim).to(self.device)
    self.target_q2 = SoftQNetwork(observation_dim,
                                  action_dim).to(self.device)
    self.target_q1.load_state_dict(self.q1.state_dict())
    self.target_q2.load_state_dict(self.q2.state_dict())

    self.policy = PolicyNetwork(observation_dim,
                                action_dim,
                                256,
                                device=self.device).to(self.device)

    self.alpha = alpha
    self.target_a = -action_dim
    self.log_a = torch.zeros(1, requires_grad=True, device=self.device)

    self.q1_criterion = nn.MSELoss()
    self.q2_criterion = nn.MSELoss()

    self.q1_optim = optim.Adam(self.q1.parameters(), lr=qlr)
    self.q2_optim = optim.Adam(self.q2.parameters(), lr=qlr)
    self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
    self.a_optim = optim.Adam([self.log_a], lr=alr)

    self.mem_size = mem_size
    self.replay_buffer = ReplayMemory(mem_size)

  def update(self, batch_size, gamma=0.99, tau=5e-3):
    """
    Update the parameters of the agent
    :param batch_size: Size of sample taken from replay memory
    :param gamma: Discount factor for calculating Q loss
    :param tau: Smoothing rate for updating target functions
    :return: None
    """

    state, action, next_state, reward, done = self.replay_buffer.sample(
        batch_size)

    # Convert all to tensors
    state = torch.FloatTensor(state).to(self.device)
    next_state = torch.FloatTensor(next_state).to(self.device)
    action = torch.FloatTensor(action).to(self.device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
    next_action, next_log_prob = self.policy.evaluate(next_state)

    # Update Q networks using the loss function
    # J(Q_i) = 1/2 (Q(s_t, a_t) - (r(s_t, a_t) + gamma * V(s_(t+1)))^2  where,
    # V(s_t) = Q(s_t, a_t) - alpha * log policy(a_t, s_t)
    next_q1 = self.target_q1(next_state, next_action)
    next_q2 = self.target_q2(next_state, next_action)
    value = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
    # print("***************************************")
    # print(reward.shape)
    # print(value.shape)
    expected_q = reward + gamma * (1 - done) * value

    q1 = self.q1(state, action)
    q2 = self.q2(state, action)
    q1_loss = self.q1_criterion(q1, expected_q.detach())
    q2_loss = self.q2_criterion(q2, expected_q.detach())
    self.q1_optim.zero_grad()
    q1_loss.backward()
    self.q1_optim.step()
    self.q2_optim.zero_grad()
    q2_loss.backward()
    self.q2_optim.step()

    # Update policy network with loss function
    # J(policy) = alpha * log policy(a_t, s_t) - Q(s_t, a_t)
    new_action, log_prob = self.policy.evaluate(state)
    policy_loss = (self.alpha * log_prob - torch.min(
        self.q1(state, new_action), self.q2(state, new_action))).mean()
    self.policy_optim.zero_grad()
    policy_loss.backward()
    self.policy_optim.step()

    # Update temperature with loss function
    # J(alpha) = -alpha * log policy(a_t, s_t) - alpha * target_alpha
    alpha_loss = (self.log_a * (-log_prob - self.target_a).detach()).mean()
    self.a_optim.zero_grad()
    alpha_loss.backward()
    self.a_optim.step()
    self.alpha = self.log_a.exp()

    # Update target networks
    # target_param = tau * param + (1 - tau) * target_param
    for param, target_param in zip(self.q1.parameters(),
                                   self.target_q1.parameters()):
      target_param.data.copy_(tau * param.data +
                              (1 - tau) * target_param.data)
    for param, target_param in zip(self.q2.parameters(),
                                   self.target_q2.parameters()):
      target_param.data.copy_(tau * param.data +
                              (1 - tau) * target_param.data)

  def save_policy(self, path):
    """
    Saves the state dictionary of the policy network the the specified path
    :param path: The path to save the state dictionary to
    :return: None
    """
    torch.save(self.policy.state_dict(), path)


def train_loop(env,
               agent,
               max_total_steps,
               max_steps,
               batch_size,
               intermediate_policies=False,
               path="./",
               verbose=False,
               update_all=True):
  """
  Training loop
  :param env: Instance of OpenAI gym environment
  :param agent: Instance of SACAgent or SAC2Agent
  :param max_total_steps: int, Maximum number of environment steps taken during training
  :param max_steps: int, Maximum number of steps in each episode
  :param batch_size: int, Size of sample taken from replay memory
  :param intermediate_policies: Bool if you want 20, 40, 60, 80% policy saved. False by default
  :param path: String, Where to save intermediate policies. './' by default
  :param verbose: Bool prints progress at 1% increments. False by default
  :param update_all: Bool whether to update after each environment step. True by default
  :return: list of rewards achieved in training
  """

  rewards = []
  steps = 0

  while steps < max_total_steps:
    state = env.reset()
    ep_reward = 0

    # Step the simulation 5 to stop learning starting midair if it is the minitaur env
    try:
      env.mdoger7
      for i in range(5):
        p.stepSimulation()
    except AttributeError:
      continue

    for step in range(max_steps):
      if verbose and not (steps % (max_total_steps // 100)):
        print("Steps: {}".format(steps))

      # Get random action until the replay memory has been filled, then get action from policy network
      if steps > 2 * batch_size:
        action = agent.policy.get_action(state).detach()
        next_state, reward, done, _ = env.step(action.numpy())
      else:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

      # FIXME : done的判断存在问题
      # pdb.set_trace()
      done = False
      # Add state action transition to replay memory
      # print('action:', action)
      # print('state:', state)
      agent.replay_buffer.push(state, action, next_state, reward, done)

      state = next_state
      ep_reward += reward
      steps += 1

      if update_all:
        if len(agent.replay_buffer) > batch_size:
          agent.update(batch_size)
      else:
        if len(agent.replay_buffer) > batch_size and not steps % 10:
          agent.update(batch_size)

      # Save the policy network at 20% increments
      if intermediate_policies and not steps % (max_total_steps // 5):
        agent.save_policy(path + "policy{}.pth".format(
            (steps // (max_total_steps // 5))))

      # Break out of loop if an end state has been reached
      if done:
        break

    rewards.append(ep_reward)

  return rewards
