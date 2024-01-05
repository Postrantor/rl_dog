# -*- coding: utf-8 -*-
"""
This file implements the gym environment of mdoger7.
"""

import os
import math
import time
from random import uniform
import importlib_metadata
# virtual
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from pybullet_utils import bullet_client as bc
from pybullet_envs.bullet.env_randomizer_base import EnvRandomizerBase

class EnvRandomizer(EnvRandomizerBase):
  """
  @brief 一个在每次重置时改变 gym 的随机器。
  """

  def __init__(self, params_list):
    self._base_mass_err = params_list['base_mass_error_range'] # -/+20%
    self._leg_mass_err = params_list['leg_mass_error_range'] # -/+20%
    self._batt_volt = params_list['battery_voltage_range'] # unit: volt
    self._viscous_damping = params_list['motor_viscous_damping_range'] # N·m·s/rad (转矩/角速度)
    self._leg_friction = params_list['leg_friction'] # 摩擦系数

  def randomize_env(self, robot):
    """
    @brief: 随机改变模型的各种物理属性
          它在每次环境重置时随机化基座、腿部的质量/惯性、足部的摩擦系数、电池电压和电机阻尼.
    @param: robot: 位于随机环境中的robot实例.
    """
    # 这种直接传入robot，然后设置的方式？不太好吧，感觉有些别扭
    body_mass_base = robot.get_base_mass_from_urdf()
    body_mass = uniform(
        body_mass_base * (1.0 + self._base_mass_err[0]),
        body_mass_base * (1.0 + self._base_mass_err[1]))
    robot.set_base_mass(body_mass)

    leg_mass_base = robot.get_leg_masses_from_urdf()
    leg_mass_lower = np.array(leg_mass_base) * (1.0 + self._leg_mass_err[0])
    leg_mass_upper = np.array(leg_mass_base) * (1.0 + self._leg_mass_err[1])
    leg_mass = [uniform(leg_mass_lower[i], leg_mass_upper[i]) for i in range(len(leg_mass_base))]
    robot.set_leg_masses(leg_mass)

    robot.set_battery_voltage(uniform(self._batt_volt[0], self._batt_volt[1]))
    robot.set_motor_viscous_damping(uniform(self._viscous_damping[0], self._viscous_damping[1]))
    robot.set_foot_friction(uniform(self._leg_friction[0], self._leg_friction[1]))


from env.robot_model import Robot

class BulletEnv(gym.Env, Robot):
  """
  @brief The gym environment for the robot.

  It simulates the locomotion of a robot, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the robot walks in 1000 steps and penalizes the energy
  expenditure.
  > 它模拟四足机器人的运动。状态空间包括所有电机和动作的角度、速度和扭矩space 是每个电机所需的电机角度。奖励函数基于 mdoger7 在 1000 步中行走多远并惩罚能量支出。
  """
  metadata = {
      "render.modes": ["human", "rgb_array"],
      "video.frames_per_second": 50
  }

  def __init__(self, params_list):
    self.params_list = params_list

    self._is_render = params_list['render']  # 是否渲染仿真
    if self._is_render:
      self._bullet_cli = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._bullet_cli = bc.BulletClient()

    self._urdf_root = params_list['urdf_env']
    self._env_randomizer = EnvRandomizer(params_list['randomizer'])

    self._time_step = params_list['time_step']
    self._cam_dist = params_list['cam_dist']
    self._cam_yaw = params_list['cam_yaw']
    self._cam_pitch = params_list['cam_pitch']
    self._env_step_counter = params_list['env_step_counter']
    self._last_frame_time = params_list['last_frame_time']
    self._num_bullet_solver_iterations = params_list['num_bullet_solver_iterations']
    self._action_bound = params_list['action_bound']
    self._last_base_position = params_list['last_base_position']
    self._observation = params_list['observation']

    self.num_substeps = params_list['num_substeps']
    self.num_motors = params_list['num_motors']
    self.motor_angle_observation_index = params_list['motor_angle_observation_index']
    self.action_eps = params_list['action_eps']
    self.observation_eps = params_list['observation_eps']
    self.render_height = params_list['render_height']
    self.render_width = params_list['render_width']
    self.motor_velocity_observation_index = self.motor_angle_observation_index + self.num_motors
    self.motor_torque_observation_index = self.motor_velocity_observation_index + self.num_motors
    self.base_orientation_observation_index = self.motor_torque_observation_index + self.num_motors

    self._action_repeat = params_list['action_repeat']  # 运动重复的次数
    self._distance_weight = params_list['distance_weight']  # 距离项在奖励中的权重
    self._energy_weight = params_list['energy_weight']  # 能量项在奖励中的权重
    self._drift_weight = params_list['drift_weight']  # 侧向漂移项在奖励中的权重
    self._shake_weight = params_list['shake_weight']  # 垂直摇晃项在奖励中的权重
    self._distance_limit = params_list['distance_limit']  # 终止episode的最大距离
    self._observation_noise_stdev = params_list['observation_noise_stdev']  # 观察噪声的标准差
    self._leg_model_enabled = params_list['leg_model_enabled']  # 是否使用腿部马达重新参数化动作空间
    self._torque_control_enabled = params_list['torque_control_enabled']  # 是否使用扭矩控制，否则使用姿态控制

    # 是否在重置时清除仿真并加载所有内容。如果设置为false，则重置只是将model放回起始位置并将其姿势设为初始配置。
    self._hard_reset = True
    hard_reset = params_list['hard_reset']

    self._pd_control_enabled = params_list['pd_control_enabled'] # 是否为每个马达启用PD控制器
    self._accurate_motor_model_enabled = params_list['accurate_motor_model_enabled'] # 是否使用准确的直流电机模型
    # PD control needs smaller time step for stability.
    if self._pd_control_enabled or self._accurate_motor_model_enabled:
      self._time_step /= self.num_substeps
      self._num_bullet_solver_iterations /= self.num_substeps
      self._action_repeat *= self.num_substeps

    #
    self.seed()
    self.reset()

    #
    observation_high = (self.robot.get_observation_upper_bound() + self.observation_eps)
    observation_low = (self.robot.get_observation_lower_bound() - self.observation_eps)
    action_dim = 12
    action_high = np.array([self._action_bound] * action_dim)

    # 这两个参数用于初始化神经网络，应该拿出去？
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

    #
    self.viewer = None
    # This assignment need to be after reset()，是必须要reset()一次？
    self._hard_reset = hard_reset

  def set_env_randomizer(self, env_randomizer):
    self._env_randomizer = env_randomizer

  def configure(self, args):
    self._args = args

  def reset(self):
    if self._hard_reset:
      self._bullet_cli.resetSimulation()
      self._bullet_cli.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
      self._bullet_cli.setTimeStep(self._time_step)
      plane = self._bullet_cli.loadURDF("%s/plane.urdf" % self._urdf_root[1])
      self._bullet_cli.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0.9])
      self._bullet_cli.configureDebugVisualizer(self._bullet_cli.COV_ENABLE_PLANAR_REFLECTION, 0)
      self._bullet_cli.setGravity(0, 0, -10)
      acc_motor = self._accurate_motor_model_enabled

      self.robot = Robot(self.params_list['robot'], bullet_cli=self._bullet_cli)
    else:
      self.robot.Reset(reload_urdf=False)

    if self._env_randomizer is not None:
      self._env_randomizer.randomize_env(self.robot)

    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._objectives = []
    self._bullet_cli.resetDebugVisualizerCamera(self._cam_dist,
                                                     self._cam_yaw,
                                                     self._cam_pitch,
                                                     [0, 0, 0])
    if not self._torque_control_enabled:
      for _ in range(100):
        if self._pd_control_enabled or self._accurate_motor_model_enabled:
          self.robot.apply_action([math.pi] * 12)
        self._bullet_cli.stepSimulation()
    return self._noisy_observation()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    """Step forward the simulation, given the action.

    Args:
      action: A list of desired motor angles for 12 motors.

    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    if self._is_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self._action_repeat * self._time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)

      base_pos = self.robot.get_base_position()
      camInfo = self._bullet_cli.getDebugVisualizerCamera()
      distance = camInfo[10]
      yaw = camInfo[8]
      pitch = camInfo[9]
      self._bullet_cli.resetDebugVisualizerCamera(distance, yaw, pitch,
                                                       base_pos)

    action = self._transform_action_to_motor_command(action)
    for _ in range(self._action_repeat):
      self.robot.apply_action(action)
      self._bullet_cli.stepSimulation()

    self._env_step_counter += 1
    reward = self._reward()
    done = self._termination()
    return np.array(self._noisy_observation()), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.robot.get_base_position()
    view_matrix = self._bullet_cli.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._bullet_cli.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(self.render_width) / self.render_height,
        nearVal=0.1,
        farVal=100.0)
    (_, _, px, _, _) = self._bullet_cli.getCameraImage(
        width=self.render_width,
        height=self.render_height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_motor_angles(self):
    """Get the mdoger7's motor angles.

    Returns:
      A numpy array of motor angles.
    """
    return np.array(self._observation[
        self.motor_angle_observation_index:self.motor_angle_observation_index +
        self.num_motors])

  def get_motor_velocities(self):
    """Get the mdoger7's motor velocities.

    Returns:
      A numpy array of motor velocities.
    """
    return np.array(self._observation[
        self.motor_velocity_observation_index:self.motor_velocity_observation_index +
        self.num_motors])

  def get_motor_torques(self):
    """Get the mdoger7's motor torques.

    Returns:
      A numpy array of motor torques.
    """
    return np.array(self._observation[
        self.motor_torque_observation_index:self.motor_torque_observation_index +
        self.num_motors])

  def get_base_orientation(self):
    """Get the mdoger7's base orientation, represented by a quaternion.

    Returns:
      A numpy array of mdoger7's orientation.
    """
    return np.array(self._observation[self.base_orientation_observation_index:])

  def is_fallen(self):
    """Decide whether the mdoger7 has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the mdoger7 is considered fallen.

    Returns:
      Boolean value that indicates whether the mdoger7 has fallen.
    """
    orientation = self.robot.get_base_orientation()
    position = self.robot.get_base_position()
    rot_mat = self._bullet_cli.getMatrixFromQuaternion(orientation)
    # print("rot_mat:", rot_mat)
    # print("Type of rot_mat:", type(rot_mat))
    # local_up_x =  rot_mat[0:3]
    # local_up_y =  rot_mat[3:6]
    local_up = rot_mat[6:]
    roll = math.atan2(rot_mat[7], rot_mat[8])
    pitch = math.asin(-rot_mat[6])
    # yaw = math.atan2(rot_mat[3], rot_mat[0])
    return np.dot(np.asarray([0, 0, 1]),
                  np.asarray(local_up)) < 0.85 or abs(roll) > 0.1 or abs(
                      pitch) > 0.2 or abs(pitch) > 0.1 or position[2] < 0.25

    # return (abs(roll) > 0.174 or abs(pitch) > 0.174 or abs(yaw) > 0.174 or pos[2] < 0.25)
    # return (np.dot(np.asarray([1, 0, 0]), np.asarray(local_up_x)) > 0.985 or np.dot(np.asarray([0, 1, 0]), np.asarray(local_up_y)) > 0.9397 or np.dot(np.asarray([0, 0, 1]), np.asarray(local_up_z)) > 0.9397 or pos[2] < 0.3)

  def get_objectives(self):
    return self._objectives

  def _termination(self):
    position = self.robot.get_base_position()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    condition = self.is_fallen() or (distance > self._distance_limit) or (
        self.robot.check_joint_contact() > 0)
    return condition

  def _reward(self):
    orientation = self.robot.get_base_orientation()
    rot_mat = self._bullet_cli.getMatrixFromQuaternion(orientation)
    local_up_x = rot_mat[0:3]
    local_up_y = rot_mat[3:6]
    local_up_z = rot_mat[6:]
    roll = math.atan2(rot_mat[7], rot_mat[8])
    # theta = - (roll **2)
    current_base_position = self.robot.get_base_position()
    forward_reward = current_base_position[0] - self._last_base_position[0]
    drift_reward = -abs(current_base_position[1])
    height_reward = -(current_base_position[2] - 0.35)**2
    xy_velocity, yaw_rate = self.robot.get_base_velocity()
    target_xy_velocity = [
        5, 5
    ]  # Define these values as per your task requirements

    # Calculate the velocity error (Euclidean distance between current velocity and target velocity)
    velocity_error = ((xy_velocity[0] - target_xy_velocity[0])**2 +
                      (xy_velocity[1] - target_xy_velocity[1])**2)**0.5
    yaw_velocity_error = (yaw_rate - 3)**2

    # Design the XY velocity reward component based on the error
    some_scale_factor = 0.1  # This is a hyperparameter that you can tune
    xy_velocity_reward = math.exp(-velocity_error / some_scale_factor)
    yaw_rate_reward = math.exp(-yaw_velocity_error / 0.1)
    # adjusted_velocity = np.abs((xy_velocity - 5) / 20)
    # adjusted_yaw = np.abs((yaw_rate - 3) / 10)
    # # Calculate the exponential of the absolute values
    # exp_abs_adjusted_velocity = 10*np.exp(adjusted_velocity)
    # exp_abs_adjusted_yaw = 10*np.exp(adjusted_yaw)
    # drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    # shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
    # shake_reward = -(20*(current_base_position[2] - 0.35)**2)
    rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
    local_up_vec = rot_matrix[6:]
    shake_reward = -abs(np.dot(np.asarray([1, 1, 0]),
                               np.asarray(local_up_vec)))
    self._last_base_position = current_base_position

    energy_reward = np.abs(
        np.dot(self.robot.get_motor_torques(),
               self.robot.get_motor_velocities())) * self._time_step
    reward = (self._distance_weight * forward_reward -
              self._energy_weight * energy_reward +
              self._drift_weight * drift_reward +
              self._shake_weight * shake_reward + 10 * xy_velocity_reward +
              10 * yaw_rate_reward + 50 * height_reward
              )  #+ self.mdoger7.CheckJointContact()
    self._objectives.append([
        forward_reward, energy_reward, drift_reward, shake_reward,
        xy_velocity_reward, yaw_rate_reward, height_reward
    ])
    return reward

  def _get_observation(self):
    self._observation = self.robot.get_observation()
    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = np.array(self._observation)
    if self._observation_noise_stdev > 0:
      observation += (np.random.normal(scale=self._observation_noise_stdev,
                                       size=observation.shape) *
                      self.robot.get_observation_upper_bound())
    return observation

  def _transform_action_to_motor_command(self, action):
    if self._leg_model_enabled:
      for i, action_component in enumerate(action):
        if not (-self._action_bound - self.action_eps <= action_component <=
                self._action_bound + self.action_eps):
          raise ValueError("{}th action {} out of bounds.".format(
              i, action_component))
      action = self.robot.convert_from_leg_model(action)
    return action

  if importlib_metadata.version('gym') < "0.9.6":
    _render = render
    _reset = reset
    _seed = seed
    _step = step
