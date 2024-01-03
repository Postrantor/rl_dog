# -*- coding: utf-8 -*-
"""
This file implements the gym environment of mdoger7.
"""

import os
import math
import time
import random
import importlib_metadata
# virtual
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data
from pybullet_envs.bullet.env_randomizer_base import EnvRandomizerBase

# env randomizer range
## relative range.
BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # -/+20%
LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # -/+20%
## absolute range
BATTERY_VOLTAGE_RANGE = (24.8, 26.8)  # unit: volt
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.1)  # N·m·s/rad (转矩/角速度)
LEG_FRICTION = (0.8, 1.5)  # 无单位(无量纲)


class EnvRandomizer(EnvRandomizerBase):
  """
  一个在每次重置时改变 gym env 的随机器。
  """

  def __init__(self,
               base_mass_err_range=BASE_MASS_ERROR_RANGE,
               leg_mass_err_range=LEG_MASS_ERROR_RANGE,
               battery_voltage_range=BATTERY_VOLTAGE_RANGE,
               motor_viscous_damping_range=MOTOR_VISCOUS_DAMPING_RANGE):
    self._base_mass_err_range = base_mass_err_range
    self._leg_mass_err_range = leg_mass_err_range
    self._battery_voltage_range = battery_voltage_range
    self._motor_viscous_damping_range = motor_viscous_damping_range

  def randomize_env(self, robot):
    """
    @brief: 随机改变模型的各种物理属性
          它在每次环境重置(`reset()`)时随机化基座、腿部的质量/惯性、足部的摩擦系数、电池电压和电机阻尼.
    @param: robot: 位于robot_gym_env环境中的robot实例.
    """
    base_mass = robot.GetBaseMassFromURDF()
    randomized_base_mass = random.uniform(
        base_mass * (1.0 + self._base_mass_err_range[0]),
        base_mass * (1.0 + self._base_mass_err_range[1]))
    robot.SetBaseMass(randomized_base_mass)

    leg_masses = robot.GetLegMassesFromURDF()
    leg_masses_lower_bound = np.array(leg_masses) * (
        1.0 + self._leg_mass_err_range[0])
    leg_masses_upper_bound = np.array(leg_masses) * (
        1.0 + self._leg_mass_err_range[1])
    randomized_leg_masses = [
        np.random.uniform(leg_masses_lower_bound[i], leg_masses_upper_bound[i])
        for i in range(len(leg_masses))
    ]
    robot.SetLegMasses(randomized_leg_masses)

    randomized_battery_voltage = random.uniform(BATTERY_VOLTAGE_RANGE[0],
                                                BATTERY_VOLTAGE_RANGE[1])
    robot.SetBatteryVoltage(randomized_battery_voltage)

    randomized_motor_damping = random.uniform(MOTOR_VISCOUS_DAMPING_RANGE[0],
                                              MOTOR_VISCOUS_DAMPING_RANGE[1])
    robot.SetMotorViscousDamping(randomized_motor_damping)

    randomized_foot_friction = random.uniform(LEG_FRICTION[0], LEG_FRICTION[1])
    robot.SetFootFriction(randomized_foot_friction)


from env.robot_model import Robot

# robot range
NUM_SUBSTEPS = 5
NUM_MOTORS = 12
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.02
OBSERVATION_EPS = 0.02
RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class BulletEnv(gym.Env):
  """
  The gym environment for the mdoger7.

  It simulates the locomotion of a mdoger7, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the mdoger7 walks in 1000 steps and penalizes the energy
  expenditure.
  它模拟四足机器人 mdoger7 的运动。 状态空间包括所有电机和动作的角度、速度和扭矩space 是每个电机所需的电机角度。 奖励函数基于mdoger7 在 1000 步中行走多远并惩罚能量支出。
  """
  metadata = {
      "render.modes": ["human", "rgb_array"],
      "video.frames_per_second": 50
  }

  def __init__(
      self,
      parameters_list,
      urdf_root=pybullet_data.getDataPath(),  # 默认值为pybullet_data的路径
      env_randomizer=EnvRandomizer(),  # 用于在reset()期间随机化物理属性的EnvRandomizer。
      # render=True,  # 是否渲染仿真
      # action_repeat=1,  # 运动重复的次数
      # distance_weight=10.0,  # 距离项在奖励中的权重
      # energy_weight=0.5,  # 能量项在奖励中的权重
      # shake_weight=5.0,  # 垂直摇晃项在奖励中的权重
      # drift_weight=5.0,  # 侧向漂移项在奖励中的权重
      # observation_noise_stdev=0.0,  # 观察噪声的标准差
      # distance_limit=float("inf"),  # 终止episode的最大距离
      # self_collision_enabled=True,  # 是否允许机器人自身碰撞
      # hard_reset=True,  # 是否在重置时清除仿真并加载所有内容。如果设置为false，则重置只是将model放回起始位置并将其姿势设为初始配置。
      # on_rack=False,  # 是否将model放在架子上。这仅用于调试行走步态。在此模式下，model的基座悬挂在半空中，以便更清晰地可视化其步态。
      # motor_velocity_limit=float("inf"),  # 每个马达的速度限制(原np.inf，修改为float("inf"))
      # pd_control_enabled=False,  # 是否为每个马达启用PD控制器
      # leg_model_enabled=True,  # 是否使用腿部马达重新参数化动作空间
      # accurate_motor_model_enabled=True,  # 是否使用准确的直流电机模型
      # torque_control_enabled=False,  # 是否使用扭矩控制，如果设置为False，则使用姿态控制
      # motor_overheat_protection=False,  # 是否关闭已施加大力矩(OVERHEAT_SHUTDOWN_TORQUE)的电机，以防止过热(OVERHEAT_SHUTDOWN_TIME)。有关更多详细信息，请参见minitaur.py中的ApplyAction()函数。
      # motor_kp=2.0,  # 准确电机模型的比例增益
      # motor_kd=0.03,  # 准确电机模型的微分增益
      # kd_for_pd_controllers=0.3,  # 用于马达的PD控制器的kd值
  ):

    self._urdf_root = urdf_root
    self._env_randomizer = env_randomizer
    self._time_step = 0.01
    self._num_bullet_solver_iterations = 300
    self._observation = []
    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._action_bound = 1
    self._cam_dist = 1.0
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._last_frame_time = 0.0
    self._is_render = parameters_list['render']
    self._action_repeat = parameters_list['action_repeat']
    self._self_collision_enabled = parameters_list['self_collision_enabled']
    self._motor_velocity_limit = parameters_list['motor_velocity_limit']
    self._distance_weight = parameters_list['distance_weight']
    self._energy_weight = parameters_list['energy_weight']
    self._drift_weight = parameters_list['drift_weight']
    self._shake_weight = parameters_list['shake_weight']
    self._distance_limit = parameters_list['distance_limit']
    self._observation_noise_stdev = parameters_list['observation_noise_stdev']
    self._leg_model_enabled = parameters_list['leg_model_enabled']
    self._motor_kp = parameters_list['motor_kp']
    self._motor_kd = parameters_list['motor_kd']
    self._torque_control_enabled = parameters_list['torque_control_enabled']
    self._motor_overheat_protection = parameters_list['motor_overheat_protection']
    self._on_rack = parameters_list['on_rack']
    self._kd_for_pd_controllers = parameters_list['kd_for_pd_controllers']

    self._hard_reset = True
    hard_reset = parameters_list['hard_reset']
    pd_control_enabled = parameters_list['pd_control_enabled']
    accurate_motor_model_enabled = parameters_list['accurate_motor_model_enabled']
    self._pd_control_enabled = parameters_list['pd_control_enabled']
    self._accurate_motor_model_enabled = parameters_list['accurate_motor_model_enabled']

    # PD control needs smaller time step for stability.
    if pd_control_enabled or accurate_motor_model_enabled:
      self._time_step /= NUM_SUBSTEPS
      self._num_bullet_solver_iterations /= NUM_SUBSTEPS
      self._action_repeat *= NUM_SUBSTEPS

    #
    if self._is_render:
      self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bc.BulletClient()

    #
    self.seed()
    self.reset()

    #
    observation_high = (self.robot.GetObservationUpperBound() +
                        OBSERVATION_EPS)
    observation_low = (self.robot.GetObservationLowerBound() - OBSERVATION_EPS)
    action_dim = 12
    action_high = np.array([self._action_bound] * action_dim)
    # 这两个参数用于初始化神经网络，应该拿出去？
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(observation_low,
                                        observation_high,
                                        dtype=np.float32)

    #
    self.viewer = None
    self._hard_reset = hard_reset  # This assignment need to be after reset()

  def set_env_randomizer(self, env_randomizer):
    self._env_randomizer = env_randomizer

  def configure(self, args):
    self._args = args

  def reset(self):
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
      self._pybullet_client.changeVisualShape(plane,
                                              -1,
                                              rgbaColor=[1, 1, 1, 0.9])
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
      self._pybullet_client.setGravity(0, 0, -10)
      acc_motor = self._accurate_motor_model_enabled
      motor_protect = self._motor_overheat_protection

      robot_urdf_path = os.path.join(os.path.dirname(__file__),
                                     '/../mdoger7/urdf/')
      self.robot = Robot(pybullet_client=self._pybullet_client,
                         urdf_root=robot_urdf_path,
                         self_collision_enabled=self._self_collision_enabled,
                         motor_velocity_limit=self._motor_velocity_limit,
                         pd_control_enabled=self._pd_control_enabled,
                         accurate_motor_model_enabled=acc_motor,
                         motor_kp=self._motor_kp,
                         motor_kd=self._motor_kd,
                         torque_control_enabled=self._torque_control_enabled,
                         motor_overheat_protection=motor_protect,
                         on_rack=self._on_rack,
                         kd_for_pd_controllers=self._kd_for_pd_controllers)
    else:
      self.robot.Reset(reload_urdf=False)

    if self._env_randomizer is not None:
      self._env_randomizer.randomize_env(self.robot)

    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._objectives = []
    self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist,
                                                     self._cam_yaw,
                                                     self._cam_pitch,
                                                     [0, 0, 0])
    if not self._torque_control_enabled:
      for _ in range(100):
        if self._pd_control_enabled or self._accurate_motor_model_enabled:
          self.robot.ApplyAction([math.pi] * 12)
        self._pybullet_client.stepSimulation()
    return self._noisy_observation()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _transform_action_to_motor_command(self, action):
    if self._leg_model_enabled:
      for i, action_component in enumerate(action):
        if not (-self._action_bound - ACTION_EPS <= action_component <=
                self._action_bound + ACTION_EPS):
          raise ValueError("{}th action {} out of bounds.".format(
              i, action_component))
      action = self.robot.ConvertFromLegModel(action)
    return action

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
      base_pos = self.robot.GetBasePosition()
      camInfo = self._pybullet_client.getDebugVisualizerCamera()
      # print("camInfo:", camInfo)
      curTargetPos = camInfo[11]
      distance = camInfo[10]
      yaw = camInfo[8]
      pitch = camInfo[9]
      targetPos = [
          0.95 * curTargetPos[0] + 0.05 * base_pos[0],
          0.95 * curTargetPos[1] + 0.05 * base_pos[1], curTargetPos[2]
      ]

      self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch,
                                                       base_pos)
    action = self._transform_action_to_motor_command(action)
    for _ in range(self._action_repeat):
      self.robot.ApplyAction(action)
      self._pybullet_client.stepSimulation()

    self._env_step_counter += 1
    reward = self._reward()
    done = self._termination()
    return np.array(self._noisy_observation()), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.robot.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
        nearVal=0.1,
        farVal=100.0)
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_mdoger7_motor_angles(self):
    """Get the mdoger7's motor angles.

    Returns:
      A numpy array of motor angles.
    """
    return np.array(self._observation[
        MOTOR_ANGLE_OBSERVATION_INDEX:MOTOR_ANGLE_OBSERVATION_INDEX +
        NUM_MOTORS])

  def get_mdoger7_motor_velocities(self):
    """Get the mdoger7's motor velocities.

    Returns:
      A numpy array of motor velocities.
    """
    return np.array(self._observation[
        MOTOR_VELOCITY_OBSERVATION_INDEX:MOTOR_VELOCITY_OBSERVATION_INDEX +
        NUM_MOTORS])

  def get_mdoger7_motor_torques(self):
    """Get the mdoger7's motor torques.

    Returns:
      A numpy array of motor torques.
    """
    return np.array(self._observation[
        MOTOR_TORQUE_OBSERVATION_INDEX:MOTOR_TORQUE_OBSERVATION_INDEX +
        NUM_MOTORS])

  def get_mdoger7_base_orientation(self):
    """Get the mdoger7's base orientation, represented by a quaternion.

    Returns:
      A numpy array of mdoger7's orientation.
    """
    return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

  def is_fallen(self):
    """Decide whether the mdoger7 has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the mdoger7 is considered fallen.

    Returns:
      Boolean value that indicates whether the mdoger7 has fallen.
    """
    orientation = self.robot.GetBaseOrientation()
    position = self.robot.GetBasePosition()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
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

  def _termination(self):
    position = self.robot.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    condition = self.is_fallen() or (distance > self._distance_limit) or (
        self.robot.CheckJointContact() > 0)
    return condition

  def _reward(self):
    orientation = self.robot.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    # print("rot_mat:", rot_mat)
    # print("Type of rot_mat:", type(rot_mat))
    local_up_x = rot_mat[0:3]
    local_up_y = rot_mat[3:6]
    local_up_z = rot_mat[6:]
    roll = math.atan2(rot_mat[7], rot_mat[8])
    # theta = - (roll **2)
    current_base_position = self.robot.GetBasePosition()
    forward_reward = current_base_position[0] - self._last_base_position[0]
    drift_reward = -abs(current_base_position[1])
    height_reward = -(current_base_position[2] - 0.35)**2
    xy_velocity, yaw_rate = self.robot.GetBaseVelocity()
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
        np.dot(self.robot.GetMotorTorques(),
               self.robot.GetMotorVelocities())) * self._time_step
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
    # print('reward:', reward)
    return reward

  def get_objectives(self):
    return self._objectives

  def _get_observation(self):
    self._observation = self.robot.GetObservation()
    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = np.array(self._observation)
    if self._observation_noise_stdev > 0:
      observation += (np.random.normal(scale=self._observation_noise_stdev,
                                       size=observation.shape) *
                      self.robot.GetObservationUpperBound())
    return observation

  if importlib_metadata.version('gym') < "0.9.6":
    _render = render
    _reset = reset
    _seed = seed
    _step = step
