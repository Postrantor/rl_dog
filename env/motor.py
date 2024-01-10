# -*- coding: utf-8 -*-
"""
本文件实现了一个准确的电机模型。
"""
import numpy as np
# plot
import matplotlib.pyplot as plt
from utils.plot_figure import PlotFigure


class MotorModel(PlotFigure):
  """
  @brief 准确的电机模型，基于直流电动机的物理原理。
    该电机模型支持两种控制类型：位置控制和力矩控制。在位置控制模式下，指定了期望的电机角度，并根据内部电机模型计算出力矩。当指定力矩控制时，将PWM信号转换为力矩。
    内部电机模型考虑以下因素：比例-微分增益、粘性摩擦、背电动势电压和电流-力矩曲线。
  """

  def __init__(self, params_list):
    self._torque_control_enabled = params_list['torque_control_enabled']
    self._kp = params_list['kp']
    self._kd = params_list['kd']
    self._resistance = params_list['motor_resistance']  # 电机电阻
    self._voltage = params_list['motor_voltage']  # 电机电压
    self._torque_constant = params_list['motor_torque_constant']  # 电机转矩常数
    self._viscous_damping = params_list['motor_viscous_damping']  # 电机粘性阻尼
    self._voltage_clipping = params_list['voltage_clipping']  # 电压限制
    self._observed_torque_limit = params_list['observed_torque_limit']  # 观测力矩限制
    self._motor_speed_limit = params_list['motor_speed_limit']  # 电机速度限制
    self._current_table = params_list['current_table']
    self._torque_table = params_list['torque_table']

  def set_voltage(self, voltage):
    self._voltage = voltage

  def get_voltage(self):
    return self._voltage

  def set_viscous_damping(self, viscous_damping):
    self._viscous_damping = viscous_damping

  def get_viscous_damping(self):
    return self._viscous_damping

  def convert_to_torque(self, motor_commands,
                        current_motor_angle,
                        current_motor_velocity):
    """
    @brief 将命令（位置控制或力矩控制）转换为力矩
    @param motor_commands: 如果电机处于位置控制模式，则是期望的电机角度。如果电机处于力矩控制模式，则是PWM信号。
    @param current_motor_angle: 当前时间步骤的电机角度。
    @param current_motor_velocity: 当前时间步骤的电机速度。
    @return actual_torque: 需要施加到电机上的力矩。
    @return observed_torque: 传感器观测到的力矩。
    """
    if self._torque_control_enabled:
      pwm = motor_commands
    else:
      pwm = (-self._kp * (current_motor_angle - motor_commands) -
             self._kd * current_motor_velocity)
    pwm = np.clip(pwm, -1.0, 1.0)
    return self._convert_to_torque_from_pwm(pwm, current_motor_velocity)

  def _convert_to_torque_from_pwm(self, pwm, current_motor_velocity):
    """
    @brief 将PWM信号转换为力矩
    @param pwm: 脉冲宽度调制信号（PWM）
    @param current_motor_velocity: 当前时间步骤的电机速度。
    @return actual_torque: 需要施加到电机上的力矩。
    @return observed_torque: 传感器观测到的力矩。
    """
    observed_torque = np.clip(
        self._torque_constant * (pwm * self._voltage / self._resistance),
        -self._observed_torque_limit,
        self._observed_torque_limit)

    # 通过电机控制器上的二极管将净电压限制在50V。
    voltage_net = np.clip(
        pwm * self._voltage - (self._torque_constant + self._viscous_damping) *
        current_motor_velocity, -self._voltage_clipping,
        self._voltage_clipping)
    current = voltage_net / self._resistance
    current_sign = np.sign(current)
    current_magnitude = np.absolute(current)

    # 根据经验电流关系对力矩进行饱和处理。
    actual_torque = np.interp(current_magnitude, self._current_table, self._torque_table)
    actual_torque = np.multiply(current_sign, actual_torque)
    return actual_torque, observed_torque
