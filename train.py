# -*- coding: utf-8 -*-
"""
@file main.py
@brief 使用强化学习算法训练智能体在BulletEnv环境中执行任务的示例代码
"""

# get logger
from sac import SAC2Agent, train_loop
from logger.logger import Logger
import env.robot_env as e

# @brief 创建BulletEnv环境对象
# @param render: 是否显示图形界面
# @param drift_weight: 漂移权重
# @param shake_weight: 摇晃权重
# @param energy_weight: 能量权重
env = e.BulletEnv(render=True,
                         drift_weight=5,
                         shake_weight=5,
                         energy_weight=0.5)

# @brief 创建SAC2Agent智能体对象
# @param env: 环境对象
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
           max_total_steps=500000,
           max_steps=1000,
           batch_size=256,
           intermediate_policies=True,
           verbose=True)

# @brief 保存最终训练得到的策略
agent.save_policy("final_policy.pth")
