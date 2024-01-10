# -*- coding: utf-8 -*-
"""
this file implements the functionalities of a minitaur using pybullet.
"""

import copy
import math
import numpy as np
# from self
from env.motor import MotorModel
# plot
import matplotlib.pyplot as plt
from utils.plot_figure import PlotFigure


class Robot(MotorModel, PlotFigure):
  """
  the robot class that simulates a quadruped robot from ghost robotics.
  """

  _observed_motor_torques = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  _applied_motor_torques = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  def __init__(self, params_list, bullet_cli):
    """
    @brief constructs a robot and reset it to the initial states.

    @param pybullet_client: the instance of bullet client to manage different simulations.
    @param urdf_root: the path to the urdf folder.
    @param time_step: the time step of the simulation.
    @param self_collision_enabled: whether to enable self collision.
    @param motor_velocity_limit: the upper limit of the motor velocity.
    @param pd_control_enabled: whether to use pd control for the motors.
    @param accurate_motor_model_enabled: whether to use the accurate dc motor model.
    @param motor_kp: proportional gain for the accurate motor model
    @param motor_kd: derivative gain for the acurate motor model
    @param torque_control_enabled: whether to use the torque control, if set to false, pose control will be used.
    @param motor_overheat_protection:
            whether to shutdown the motor that has exerted large torque (overheat_shutdown_torque) for an extended amount of time (overheat_shutdown_time). see applyaction() in mdoger7.py for more details.
    @param on_rack:
            whether to place the mdoger7 on rack. this is only used to debug the walking gait. in this mode, the mdoger7's base is hanged midair so that its walking gait is clearer to visualize.
    @param kd_for_pd_controllers: kd value for the pd controllers of the motors.
    """
    self._bullet_cli = bullet_cli
    self._parse_config(params_list)
    self.reset()

  def _parse_config(self, params_list):
    self._on_rack = params_list['on_rack']

    self.num_motors = params_list['num_motors']
    self.num_legs = params_list['num_legs']

    self._time_step = params_list['time_step']

    self._self_collision_enabled = params_list['self_collision_enabled']
    self._max_force = params_list['max_force']

    self.lower_constraint_point_right = params_list['lower_constraint_point_right']
    self.lower_constraint_point_left = params_list['lower_constraint_point_left']

    self._set_params_motor(params_list)
    self._set_params_link_id(params_list)
    self._set_params_urdf(params_list)
    self._set_params_kb_kd(params_list)
    self._set_params_robot_position(params_list)

  def _set_params_robot_position(self, params_list):
    # bases on the readings from 's default pose.
    self.motor_direction = params_list['init_position']['motor_direction']
    self.init_position = params_list['init_position']['init_position']
    self.init_orientation = params_list['init_position']['init_orientation']
    self.init_motor_angles = self.num_legs * [
        params_list['init_position']['default_abduction_angle'],
        params_list['init_position']['default_hip_angle'],
        params_list['init_position']['default_knee_angle']]

  def _set_params_kb_kd(self, params_list):
    self._pd_control_enabled = params_list['pd_control_enabled']
    self._accurate_motor_model_enabled = params_list['accurate_motor_model_enabled']

    if self._accurate_motor_model_enabled:
      self._motor_model = MotorModel(params_list['motor'])
      self._kp = params_list['motor']['kp']
      self._kd = params_list['motor']['kd']
    elif self._pd_control_enabled:
      self._kp = 1
      self._kd = params_list['kd_for_pd_controllers']
    else:
      self._kp = 1
      self._kd = 1

  def _set_params_urdf(self, params_list):
    self._urdf_robot = params_list['urdf_env'][0]
    self._urdf_env = params_list['urdf_env'][1]

  def _set_params_link_id(self, params_list):
    self.link_names = params_list['link']['link_names']
    self.base_link_id = params_list['link']['base_id']
    self.foot_link_id = params_list['link']['foot_id']
    self.motor_link_id = params_list['link']['motor_id']

  def _set_params_motor(self, params_list):
    """
    if OVERHEAT_SHUTDOWN_TIME, shutdown motor
    请参见minitaur.py中的ApplyAction()函数。
    """
    # overheat
    self.motor_overheat_protection = params_list['motor']['overheat']['protection']
    self.overheat_shutdown_torque = params_list['motor']['overheat']['shutdown_torque']
    self.overheat_shutdown_time = params_list['motor']['overheat']['shutdown_time']
    # limit
    self.joint_velocity = params_list['motor']['motor_speed_limit']
    self.joint_torque = params_list['motor']['observed_torque_limit']

  def reset(self, reload_urdf=True):
    """
    @brief: reset the robot to its initial states.
    @param: reload_urdf: whether to reload the urdf file.
    """
    # load urdf
    self._plant = self._bullet_cli.loadURDF("%s/plane.urdf" % self._urdf_env)
    self._bullet_cli.changeVisualShape(self._plant, -1, rgbaColor=[1, 1, 1, 0.9])
    # load robot
    if reload_urdf:
      if self._self_collision_enabled:
        self.quadruped = self._bullet_cli.loadURDF(
            self._urdf_robot,
            self.init_position,
            self.init_orientation,
            useFixedBase=self._on_rack,
            flags=self._bullet_cli.URDF_USE_SELF_COLLISION)
      else:
        self.quadruped = self._bullet_cli.loadURDF(
            self._urdf_robot,
            self.init_position,
            self.init_orientation,
            useFixedBase=self._on_rack)
      self._build_joint_name2id_dict()
      self._build_motor_id_list()
      self._record_mass_info_from_urdf()
      self.reset_pose()
    else:
      self._bullet_cli.resetBasePositionAndOrientation(self.quadruped,
                                                       self.init_position,
                                                       self.init_orientation)
      self._bullet_cli.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
      self.reset_pose()

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors

  # FIXME(@zhiqi.jia) need add add_constraint
  # ~\anaconda3\envs\pytorch\Lib\site-packages\pybullet_envs\bullet\minitaur.py
  def reset_pose(self):
    for name, i in zip(self.link_names, range(len(self.link_names))):
      angle = self.init_motor_angles[i]
      self._bullet_cli.resetJointState(self.quadruped,
                                       self._joint_name_to_id[name],
                                       angle,
                                       targetVelocity=0)
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._bullet_cli.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._bullet_cli.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)

  def _record_mass_info_from_urdf(self):
    self._base_mass_urdf = self._bullet_cli.getDynamicsInfo(self.quadruped, self.base_link_id)[0]
    self._leg_masses_urdf = []
    self._leg_masses_urdf.append(self._bullet_cli.getDynamicsInfo(self.quadruped, self.motor_link_id[0])[0])

  def _build_joint_name2id_dict(self):
    num_joints = self._bullet_cli.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._bullet_cli.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("utf-8")] = joint_info[0]

  def _build_motor_id_list(self):
    self._motor_id_list = [self._joint_name_to_id[motor_name]
                           for motor_name in self.link_names]

  def _set_motor_torque_by_id(self, motor_id, torque, enable=True):
    if not enable:
      torque = 0
    self._bullet_cli.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._bullet_cli.TORQUE_CONTROL,
        force=torque)

  def _set_desired_motor_angle_by_name(self, motor_name, desired_angle):
    self._set_desired_motor_angle_by_id(self._joint_name_to_id[motor_name], desired_angle)

  def _set_desired_motor_angle_by_id(self, motor_id, desired_angle):
    self._bullet_cli.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._bullet_cli.POSITION_CONTROL,
        targetPosition=desired_angle,
        positionGain=self._kp,
        velocityGain=self._kd,
        force=self._max_force)

  def check_joint_contact(self):
    """
    @brief: 检查指定的连杆是否与地面接触
    @return: 是否与地面接触
    """
    # 初始化接触检测结果
    collision_count = False
    # 转换为连杆 ID
    motor_joint_ids = [self._joint_name_to_id[name] for name in self.link_names]
    # 获取所有接触点
    contact_points = self._bullet_cli.getContactPoints(
        bodyA=self._plant,
        bodyB=self.quadruped)
    # 检查接触
    for contact in contact_points:
      link_id = contact[4]  # 接触的连杆 ID
      if link_id in motor_joint_ids:
        collision_count = True
    return collision_count

  def get_base_position(self):
    """
    @brief get the position of robot's base.
    @return: the position of robot's base.
    """
    position, _ = (self._bullet_cli.getBasePositionAndOrientation(self.quadruped))
    return position

  def get_base_orientation(self):
    """
    @brief get the orientation of robot's base, represented as quaternion.
    @return: the orientation of robot's base.
    """
    _, orientation = (self._bullet_cli.getBasePositionAndOrientation(self.quadruped))
    return orientation

  def get_base_velocity(self):
    """
    @brief get the velocity of robot's base in the xy plane and the yaw rate.
    @return: a tuple containing:
      - the xy velocity of robot's base.
      - the yaw rate (rotational velocity around z-axis) of robot's base.
    """
    linear_velocity, angular_velocity = self._bullet_cli.getBaseVelocity(self.quadruped)
    xy_velocity = np.array(linear_velocity[:2])  # take only the x and y components
    yaw_rate = angular_velocity[2]  # z-axis component represents yaw rate
    return xy_velocity, yaw_rate

  def get_action_dimension(self):
    """
    @brief get the length of the action list.
    @return: the length of the action list.
    """
    return self.num_motors

  def get_observation_upper_bound(self):
    """
    @brief get the upper bound of the observation.
    @return the upper bound of an observation. see getobservation() for the details of each element of an observation.
    """
    # FIXME(@zhiqi.jia) 这里设置的上/下限范围不太合理
    upper_bound = np.array([0.0] * self.GetObservationDimension())
    upper_bound[0:self.num_motors] = math.pi  # Joint angle.
    upper_bound[self.num_motors:2 * self.num_motors] = self.joint_velocity
    upper_bound[2 * self.num_motors:3 * self.num_motors] = self.joint_torque
    upper_bound[3 * self.num_motors:3 * self.num_motors + 4] = 1.0  # quaternion of base orientation.

    # assuming a reasonable upper limit for base xy velocity (e.g., 10 m/s)
    # upper limit for XY velocity
    upper_bound[3 * self.num_motors + 4:3 * self.num_motors + 6] = 10.0

    # assuming a reasonable upper limit for yaw rate (e.g., 5 rad/s)
    # upper limit for yaw rate
    upper_bound[3 * self.num_motors + 6] = 5.0

    return upper_bound

  def get_observation_lower_bound(self):
    """get the lower bound of the observation."""
    return -self.get_observation_upper_bound()

  def GetObservationDimension(self):
    """
    @brief get the length of the observation list.
    @return the length of the observation list.
    """
    return len(self.get_observation())

  def get_observation(self):
    """
    @brief get the observations of mdoger.
      it includes the angles, velocities, torques and the orientation of the base.

    @return the observation list. observation[0:12] are motor angles. observation[12:24] are motor velocities, observation[24:36] are motor torques. observation[36:48] is the orientation of the base, in quaternion form.
    """
    observation = []
    observation.extend(self.get_motor_angles().tolist())
    observation.extend(self.get_motor_velocities().tolist())
    observation.extend(self.get_motor_torques().tolist())
    observation.extend(list(self.get_base_orientation()))

    xy_velocity, yaw_rate = self.get_base_velocity()
    observation.extend(xy_velocity.tolist())  # add xy velocity to the observation
    observation.append(yaw_rate)  # add yaw rate to the observation
    return observation

  def apply_action(self, motor_cmds):
    """
    @brief set the desired motor angles to the motors of the mdoger.
      the desired motor angles are clipped based on the maximum allowed velocity. if the pd_control_enabled is true, a torque is calculated according to the difference between current and desired joint angle, as well as the joint velocity. this torque is exerted to the motor. for more information about pd control, please refer to: https://en.wikipedia.org/wiki/pid_controller.
    @param motor_commands: the 12 desired motor angles.
    """
    if self.joint_velocity < np.inf:
      cur_motor_angle = self.get_motor_angles()
      motor_cmd_max = (cur_motor_angle + self._time_step * self.joint_velocity)
      motor_cmd_min = (cur_motor_angle - self._time_step * self.joint_velocity)
      motor_cmds = np.clip(motor_cmds, motor_cmd_min, motor_cmd_max)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      q = self.get_motor_angles()
      qdot = self.get_motor_velocities()
      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(motor_cmds, q, qdot)
        # 电机过热保护，超过时间阈值则关闭电机
        if self.motor_overheat_protection:
          for i in range(self.num_motors):
            if abs(actual_torque[i]) > self.overheat_shutdown_torque:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] = 0
            if (self._overheat_counter[i] > self.overheat_shutdown_time / self._time_step):
              self._motor_enabled_list[i] = False

        # the torque is already in the observation space because we use
        # getmotorangles and getmotorvelocities.
        self._observed_motor_torques = observed_torque

        # transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque, self.motor_direction)
        for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
                                                         self._applied_motor_torque,
                                                         self._motor_enabled_list):
          self._set_motor_torque_by_id(motor_id, motor_torque, motor_enabled)
      else:
        torque_commands = -self._kp * (q - motor_cmds) - self._kd * qdot
        # the torque is already in the observation space because we use
        # getmotorangles and getmotorvelocities.
        self._observed_motor_torques = torque_commands
        # transform into the motor space when applying the torque.
        self._applied_motor_torques = np.multiply(self._observed_motor_torques,
                                                  self.motor_direction)
        for motor_id, motor_torque in zip(self._motor_id_list,
                                          self._applied_motor_torques):
          self._set_motor_torque_by_id(motor_id, motor_torque)
    else:
      motor_commands_with_direction = np.multiply(motor_cmds, self.motor_direction)
      for motor_id, motor_command_with_direction in zip(
              self._motor_id_list, motor_commands_with_direction):
        self._set_desired_motor_angle_by_id(motor_id, motor_command_with_direction)

  def get_motor_angles(self):
    """
    @brief get the eight motor angles at the current moment.
    @return motor angles.
    """
    motor_angles = [self._bullet_cli.getJointState(self.quadruped, motor_id)[0]
                    for motor_id in self._motor_id_list]
    motor_angles = np.multiply(motor_angles, self.motor_direction)
    return motor_angles

  def get_motor_velocities(self):
    """
    @brief get the velocity of all 12 motors.
    @return velocities of all 12 motors.
    """
    motor_velocities = [self._bullet_cli.getJointState(self.quadruped, motor_id)[1]
                        for motor_id in self._motor_id_list]
    motor_velocities = np.multiply(motor_velocities, self.motor_direction)
    return motor_velocities

  def get_motor_torques(self):
    """
    @brief get the amount of torques the motors are exerting.
    @return motor torques of all 12 motors.
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [self._bullet_cli.getJointState(self.quadruped, motor_id)[3]
                       for motor_id in self._motor_id_list]
      motor_torques = np.multiply(motor_torques, self.motor_direction)
    return motor_torques

  # FIXME(@zhiqi.jia) from minitaur
  def convert_from_leg_model(self, actions):
    motor_angle = copy.deepcopy(actions)
    for i in range(len(actions)):
      motor_angle[i] = math.fmod(actions[i], math.pi)
      if motor_angle[i] >= math.pi / 2:
        motor_angle[i] -= math.pi
      elif motor_angle[i] < -math.pi:
        motor_angle[i] += math.pi
    return motor_angle

  def get_base_mass_from_urdf(self):
    """get the mass of the base from the urdf file."""
    return self._base_mass_urdf

  def get_leg_masses_from_urdf(self):
    """get the mass of the legs from the urdf file."""
    return self._leg_masses_urdf

  def set_base_mass(self, base_mass):
    self._bullet_cli.changeDynamics(self.quadruped, self.base_link_id, mass=base_mass)

  def set_leg_masses(self, leg_masses):
    """
    @brief set the mass of the legs.
      a leg includes leg_link and motor. all four leg_links have the same mass, which is leg_masses[0]. all four motors have the same mass, which is leg_mass[1].
    @param leg_masses: the leg masses. leg_masses[0] is the mass of the leg link.
            leg_masses[1] is the mass of the motor.
    """
    for link_id in self.motor_link_id:
      self._bullet_cli.changeDynamics(self.quadruped, link_id, mass=leg_masses[0])

  def set_foot_friction(self, foot_friction):
    """
    @brief Set the lateral friction of the feet.
    @param foot_friction: The lateral friction coefficient of the foot. This value is shared by all four feet.
    """
    for link_id in self.foot_link_id:
      self._bullet_cli.changeDynamics(self.quadruped,
                                      link_id,
                                      lateralFriction=foot_friction)

  def set_battery_voltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def set_motor_viscous_damping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)
