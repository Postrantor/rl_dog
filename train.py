# -*- coding: utf-8 -*-
"""
@file main.py
@brief 使用强化学习算法训练智能体在BulletEnv环境中执行任务的示例代码
"""

from sac import SAC2Agent, train_loop
from logger.logger import Logger  # get logger
import env.robot_env as e
from utils import load_yaml


def algorithm_rl(parameter_list):

  ## @brief 创建BulletEnv环境对象
  # @param render: 是否显示图形界面
  # @param drift_weight: 漂移权重
  # @param shake_weight: 摇晃权重
  # @param energy_weight: 能量权重
  env = e.BulletEnv(render=parameter_list['render'],
                    drift_weight=parameter_list['drift_weight'],
                    shake_weight=parameter_list['shake_weight'],
                    energy_weight=parameter_list['energy_weight'])

  ## @brief 创建SAC2Agent智能体对象
  agent = SAC2Agent(env)

  # @brief 使用训练循环函数进行训练
  # @param env: 环境对象
  # @param agent: 智能体对象
  # @param num_episodes: 训练的总轮数
  # @param max_timesteps: 每轮训练的最大时间步数
  # @param batch_size: 批次大小
  # @param intermediate_policies: 是否保存每个训练阶段的中间策略
  # @param verbose: 是否打印训练过程中的详细信息
  train_loop(env,
             agent,
             max_total_steps=parameter_list['max_total_steps'],
             max_steps=parameter_list['max_steps'],
             batch_size=parameter_list['batch_size'],
             intermediate_policies=parameter_list['intermediate_policies'],
             verbose=parameter_list['verbose'])

  # @brief 保存最终训练得到的策略
  agent.save_policy("final_policy.pth")


def main():
  parameters = load_yaml.load_parameters_from_yaml('config/parameters.yaml')
  algorithm_rl(parameters['training'])


if __name__ == "__main__":
  main()
