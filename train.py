# -*- coding: utf-8 -*-
"""
@file main.py
@brief 使用强化学习算法训练智能体在BulletEnv环境中执行任务的示例代码
"""

from algo.sac import SAC2Agent, train_loop
from logger.logger import Logger  # get logger
import env.robot_env as e
from utils import load_yaml


def algorithm_rl(parameter_list):

  # 创建BulletEnv环境对象及Agent对象
  env = e.BulletEnv(parameter_list['environment'])
  agent = SAC2Agent(env)

  # 使用训练循环函数进行训练
  train_loop(env, agent, parameter_list['training'])
  agent.save_policy(parameter_list['final_policy'])


def main():
  parameters = load_yaml.load_parameters_from_yaml('config/parameters.yaml')
  algorithm_rl(parameters)


if __name__ == "__main__":
  main()
