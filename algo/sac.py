# -*- coding: utf-8 -*-

import os
import numpy as np
import pybullet as p
#
import torch
import torch.nn as nn
import torch.optim as optim
from network.network import SoftQNetwork
from network.network import ValueNetwork
from network.network import PolicyNetwork
# from self
from logger.logger import Logger  # get logger
from utils.replay_memory import ReplayMemory

logger = Logger().get_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SAC2Agent:
  """
  Agent for the second generation of the Soft Actor Critic learning algorithm presented in
  Soft Actor-Critic Algorithms and Applications, Haarnoja et al. 2018
  
  Differs from SACAgent by replacing the need for a value function with an entropy temperature parameter and tuning this while learning
  > 与SACAgent不同之处在于，它通过消除对值函数的需求，并使用熵温度参数来进行调整
  """

  def __init__(self, env, parameters_list):
    """
    :param env: OpenAI Gym环境的实例，模型将在其上进行学习
    :param alpha: 初始alpha值，float类型
    :param alr: 用于更新alpha的学习率，float类型
    :param qlr: 用于更新q函数的学习率，float类型
    :param policy_lr: 用于更新策略函数的学习率，float类型
    :param mem_size: 回放缓冲区的大小，int类型，默认为1e6
    """

    alpha = parameters_list['alpha']
    alr = parameters_list['alr']
    qlr = parameters_list['qlr']
    policy_lr = parameters_list['policy_lr']
    mem_size = parameters_list['mem_size']

    try:
      action_dim = env.action_space.shape[0]
    except IndexError:
      action_dim = env.action_space.n
    try:
      observation_dim = env.observation_space.shape[0]
    except IndexError:
      observation_dim = env.observation_space.n

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.q1 = SoftQNetwork(observation_dim, action_dim).to(self.device)
    self.q2 = SoftQNetwork(observation_dim, action_dim).to(self.device)

    self.target_q1 = SoftQNetwork(observation_dim, action_dim).to(self.device)
    self.target_q2 = SoftQNetwork(observation_dim, action_dim).to(self.device)
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
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for param, target_param in zip(self.q2.parameters(),
                                   self.target_q2.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  def save_policy(self, path):
    """
    Saves the state dictionary of the policy network the the specified path
    :param path: The path to save the state dictionary to
    :return: None
    """
    torch.save(self.policy.state_dict(), path)


def train_loop(env, agent, parameters_list):
  """
  训练循环
  :param env: OpenAI gym环境的实例
  :param agent: SACAgent或SAC2Agent的实例
  :param max_total_steps: int, 训练过程中环境步骤的最大值
  :param max_steps: int, 每个episode的最大步骤数
  :param batch_size: int, 从回放记忆中提取的样本大小
  :param intermediate_policies: Bool，是否保存20%，40%，60%，80%策略。默认为False
  :param path: String, 保存中间策略的路径，默认为'./'
  :param verbose: Bool，以1%的增量打印进度。默认为False
  :param update_all: Bool，是否在每个环境步骤后更新。默认为True
  :return: 训练过程中获得的奖励列表
  """

  max_total_steps = parameters_list['max_total_steps']
  max_steps = parameters_list['max_steps']
  batch_size = parameters_list['batch_size']
  path = parameters_list['path']
  intermediate_policies = parameters_list['intermediate_policies']
  verbose = parameters_list['verbose']
  update_all = parameters_list['update_all']

  #
  rewards = []
  steps = 0

  while steps < max_total_steps:
    state = env.reset()
    ep_reward = 0

    # Step the simulation 5 to stop learning starting midair if it is the minitaur env
    try:
      env.robot
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
        agent.save_policy(path +
                          "policy{}.pth".format((steps //
                                                 (max_total_steps // 5))))

      # Break out of loop if an end state has been reached
      if done:
        break

    rewards.append(ep_reward)

  return rewards
