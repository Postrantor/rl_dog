# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SoftQNetwork(nn.Module):
  """
  Soft Q Neural Network
  """
  def __init__(self, state_dim, action_dim, hidden_size=256, init_w=3e-3):
    super(SoftQNetwork, self).__init__()

    self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, 1)

    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)

  def forward(self, state, action):
    """
    Feeds a state and action forward through the soft Q network
    :param state: state to be fed through the network, float tensor
    :param action: action to be fed through the network, float tensor
    :return: Q value of the state and action, float
    """
    x = torch.cat([state, action], 1)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x


class ValueNetwork(nn.Module):
  """
  Value Network
  Only used in SACAgent, replaced by entropy temperature in SAC2Agent
  """
  def __init__(self, state_dim, hidden_dim, init_w=3e-3):
    super(ValueNetwork, self).__init__()

    self.linear1 = nn.Linear(state_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = nn.Linear(hidden_dim, 1)

    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)

  def forward(self, state):
    """
    Feeds a state forward through the value network
    :param state: state to be fed through the network, float tensor
    :return: value of the state , float
    """
    x = F.relu(self.linear1(state))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x


class PolicyNetwork(nn.Module):
  """
  Gaussian Policy network
  """
  def __init__(self,
               state_dim,
               action_dim,
               hidden_size,
               init_w=3e-3,
               log_std_min=-20,
               log_std_max=2,
               device="cpu"):
    super(PolicyNetwork, self).__init__()

    self.log_std_min = log_std_min
    self.log_std_max = log_std_max

    self.linear1 = nn.Linear(state_dim, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)

    self.mean_linear = nn.Linear(hidden_size, action_dim)
    self.mean_linear.weight.data.uniform_(-init_w, init_w)
    self.mean_linear.bias.data.uniform_(-init_w, init_w)

    self.log_std_linear = nn.Linear(hidden_size, action_dim)
    self.log_std_linear.weight.data.uniform_(-init_w, init_w)
    self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    self.device = device

  def forward(self, state):
    """
    Feeds a state forward through the policy network
    :param state: state to be fed through the network, float tensor
    :return: mean and standard deviation of the probability distribution of action given state, tensor
    """
    x = F.relu(self.linear1(state))
    x = F.relu(self.linear2(x))

    mean = self.mean_linear(x)
    log_std = self.log_std_linear(x)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

    return mean, log_std

  def evaluate(self, state, epsilon=1e-6):
    """
    Get an action and the log of the probability of that action given state
    Used for calculating loss functions
    :param state: Environment state, float tensor
    :param epsilon: noise
    :return: action: environment action, float tensor
        log_prob: log(pi(action | state)), float
    """
    mean, log_std = self.forward(state)
    std = log_std.exp()

    # Sample an action from the gaussian distribution with the mean and std
    normal = Normal(0, 1)
    z = normal.sample()
    action = torch.tanh(mean + std * z.to(self.device))

    # Get the log of the probability of action plus some noise
    log_prob = Normal(
        mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(
            1 - action.pow(2) + epsilon)

    return action, log_prob

  def get_action(self, state):
    """
    Get an action given state. Used in training
    :param state: environment state float tensor
    :return: action: float tensor
    """
    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    mean, log_std = self.forward(state)
    std = log_std.exp()

    # Sample an action from the gaussian distribution with the mean and std
    normal = Normal(0, 1)
    z = normal.sample().to(self.device)
    action = torch.tanh(mean + std * z)

    action = action.cpu()
    return action[0]
