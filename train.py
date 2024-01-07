# -*- coding: utf-8 -*-
"""
@brief 使用强化学习算法训练智能体在BulletEnv环境中执行任务
"""

from algo.sac import SAC2Agent, train_loop
from logger.logger import Logger  # get logger
import env.robot_env as e
from utils.load_yaml import load_parameters


def algorithm_rl(parameter_list):
  # 创建BulletEnv环境对象及Agent对象
  env = e.BulletEnv(parameter_list['environment'])
  agent = SAC2Agent(env, parameter_list['sac'])

  # 使用训练循环函数进行训练
  train_loop(env, agent, parameter_list['training'])
  agent.save_policy(parameter_list['training']['final_policy'])


if __name__ == "__main__":
  algorithm_rl(load_parameters('config/parameters.yaml'))
