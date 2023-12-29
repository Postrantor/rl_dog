# -*- coding: utf-8 -*-

import random
import numpy as np
from pybullet_envs.bullet import env_randomizer_base

# Relative range.
mdoger7_BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 表示 20%
mdoger7_LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 表示 20%
# 绝对范围。
BATTERY_VOLTAGE_RANGE = (24.8, 26.8)  # 单位：伏特(Volt)
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.1)  # 单位：牛顿*米*秒/弧度 (转矩/角速度)
mdoger7_LEG_FRICTION = (0.8, 1.5)  # 无单位(无量纲)


class mdoger7EnvRandomizer(env_randomizer_base.EnvRandomizerBase):
  """一个在每次重置(reset())时改变 mdoger7_gym_env 的随机器。"""
  def __init__(self,
               mdoger7_base_mass_err_range=mdoger7_BASE_MASS_ERROR_RANGE,
               mdoger7_leg_mass_err_range=mdoger7_LEG_MASS_ERROR_RANGE,
               battery_voltage_range=BATTERY_VOLTAGE_RANGE,
               motor_viscous_damping_range=MOTOR_VISCOUS_DAMPING_RANGE):
    self._mdoger7_base_mass_err_range = mdoger7_base_mass_err_range
    self._mdoger7_leg_mass_err_range = mdoger7_leg_mass_err_range
    self._battery_voltage_range = battery_voltage_range
    self._motor_viscous_damping_range = motor_viscous_damping_range

  def randomize_env(self, env):
    self._randomize_mdoger7(env.mdoger7)

  def _randomize_mdoger7(self, mdoger7):
    """随机改变mdoger7的各种物理属性

        它在每次环境重置(reset())时随机化基座、腿部的质量/惯性、足部的摩擦系数、电池电压和电机阻尼.

        Args:
            mdoger7: 位于mdoger7_gym_env环境中的mdoger7实例.
        """
    base_mass = mdoger7.GetBaseMassFromURDF()
    randomized_base_mass = random.uniform(
        base_mass * (1.0 + self._mdoger7_base_mass_err_range[0]),
        base_mass * (1.0 + self._mdoger7_base_mass_err_range[1]))
    mdoger7.SetBaseMass(randomized_base_mass)

    leg_masses = mdoger7.GetLegMassesFromURDF()
    leg_masses_lower_bound = np.array(leg_masses) * (
        1.0 + self._mdoger7_leg_mass_err_range[0])
    leg_masses_upper_bound = np.array(leg_masses) * (
        1.0 + self._mdoger7_leg_mass_err_range[1])
    randomized_leg_masses = [
        np.random.uniform(leg_masses_lower_bound[i], leg_masses_upper_bound[i])
        for i in range(len(leg_masses))
    ]
    mdoger7.SetLegMasses(randomized_leg_masses)

    randomized_battery_voltage = random.uniform(BATTERY_VOLTAGE_RANGE[0],
                                                BATTERY_VOLTAGE_RANGE[1])
    mdoger7.SetBatteryVoltage(randomized_battery_voltage)

    randomized_motor_damping = random.uniform(MOTOR_VISCOUS_DAMPING_RANGE[0],
                                              MOTOR_VISCOUS_DAMPING_RANGE[1])
    mdoger7.SetMotorViscousDamping(randomized_motor_damping)

    randomized_foot_friction = random.uniform(mdoger7_LEG_FRICTION[0],
                                              mdoger7_LEG_FRICTION[1])
    mdoger7.SetFootFriction(randomized_foot_friction)
