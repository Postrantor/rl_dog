# -*- coding: utf-8 -*-
"""
This file implements the functionalities of a minitaur using pybullet.
"""

import copy
import math
import numpy as np
# from self
from env import motor

default_abduction_angle = 0.0
default_hip_angle = 0
default_knee_angle = 0
overheat_shutdown_torque = 2.45
overheat_shutdown_time = 1.0
lower_constraint_point_right = [0, 0.00, 0.]
lower_constraint_point_left = [0, 0.0, 0.]

leg_position = ["lf", "rf", "lb", "rb"]
motor_names = [
    "lf1_joint", "lf2_joint", "lf3_joint",
    "rf1_joint", "rf2_joint", "rf3_joint",
    "lb1_joint", "lb2_joint", "lb3_joint",
    "rb1_joint", "rb2_joint", "rb3_joint"]

# bases on the readings from 's default pose.
init_position = [0, 0, 0.3]
init_orientation = [0, 0, 0, 1]
# init_motor_angles = 0*[1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]
num_legs = 4
init_motor_angles = num_legs * [
    default_abduction_angle,
    default_hip_angle,
    default_knee_angle]

motor_link_id = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
foot_link_id = [3, 7, 11, 115]
base_link_id = -1


class Robot():
  """
  The robot class that simulates a quadruped robot from Ghost Robotics.
  """

  def __init__(self, params_list, bullet_cli):
    """
    @brief constructs a robot and reset it to the initial states.

    @param pybullet_client: the instance of bulletclient to manage different simulations.
    @param urdf_root: the path to the urdf folder.
    @param time_step: the time step of the simulation.
    @param self_collision_enabled: whether to enable self collision.
    @param motor_velocity_limit: the upper limit of the motor velocity.
    @param pd_control_enabled: whether to use pd control for the motors.
    @param accurate_motor_model_enabled: whether to use the accurate dc motor model.
    @param motor_kp: proportional gain for the accurate motor model
    @param motor_kd: derivative gain for the acurate motor model
    @param torque_control_enabled: whether to use the torque control, if set to false, pose control will be used.
    @param motor_overheat_protection: whether to shutdown the motor that has exerted large torque (overheat_shutdown_torque) for an extended amount of time (overheat_shutdown_time). see applyaction() in mdoger7.py for more details.
    @param on_rack: whether to place the mdoger7 on rack. this is only used to debug the walking gait. in this mode, the mdoger7's base is hanged midair so that its walking gait is clearer to visualize.
    @param kd_for_pd_controllers: kd value for the pd controllers of the motors.
    """
    # 通过一个get_()获取？
    self._pybullet_client = bullet_cli

    self.parameters_list = params_list
    self.num_motors = params_list['num_motors']
    self.num_legs = params_list['num_legs']
    self._urdf_env = params_list['urdf_env'][1]
    self._urdf_robot = params_list['urdf_env'][0]
    self._self_collision_enabled = params_list['self_collision_enabled']
    self._motor_velocity_limit = params_list['motor_velocity_limit']
    self._pd_control_enabled = params_list['pd_control_enabled']
    self._accurate_motor_model_enabled = params_list['accurate_motor_model_enabled']
    self._motor_overheat_protection = params_list['motor_overheat_protection']
    self._on_rack = params_list['on_rack']
    self.time_step = params_list['time_step']
    self._motor_direction = [-1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1]
    self._h1motor_direction = [-1, -1, 1, 1]
    self._h2motor_direction = [-1, 1, 1, -1]
    self._h3motor_direction = [1, -1, -1, 1]
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 15

    if self._accurate_motor_model_enabled:
      self._motor_model = motor.MotorModel(params_list['motor'])
      self._kp = params_list['motor']['kp']
      self._kd = params_list['motor']['kd']
    elif self._pd_control_enabled:
      self._kp = 1
      self._kd = params_list['kd_for_pd_controllers']
    else:
      self._kp = 1
      self._kd = 1

    self.ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_env)

    self.Reset()

  def _record_mass_info_from_urdf(self):
    self._base_mass_urdf = self._pybullet_client.getDynamicsInfo(
        self.quadruped, base_link_id)[0]
    self._leg_masses_urdf = []
    # self._leg_masses_urdf.append(
    #     self._pybullet_client.getDynamicsInfo(self.quadruped, LEG_LINK_ID[0])[0])
    self._leg_masses_urdf.append(
        self._pybullet_client.getDynamicsInfo(self.quadruped,
                                              motor_link_id[0])[0])

  def _build_joint_name2id_dict(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _build_motor_id_list(self):
    self._motor_id_list = [
        self._joint_name_to_id[motor_name] for motor_name in motor_names
    ]

  def _set_motor_torque_by_id(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        force=torque)

  def _set_desired_motor_angle_by_id(self, motor_id, desired_angle):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.POSITION_CONTROL,
        targetPosition=desired_angle,
        positionGain=self._kp,
        velocityGain=self._kd,
        force=self._max_force)

  def _set_desired_motor_angle_by_name(self, motor_name, desired_angle):
    self._set_desired_motor_angle_by_id(self._joint_name_to_id[motor_name],
                                   desired_angle)

  def Reset(self, reload_urdf=True):
    """
    @brief: reset the mdoger7 to its initial states.
    @param: reload_urdf: whether to reload the urdf file.
    """

    if reload_urdf:
      if self._self_collision_enabled:
        self.quadruped = self._pybullet_client.loadURDF(
            self._urdf_robot,
            init_position,
            init_orientation,
            useFixedBase=self._on_rack,
            flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
      else:
        self.quadruped = self._pybullet_client.loadURDF(
            self._urdf_robot,
            init_position,
            init_orientation,
            useFixedBase=self._on_rack)
      self._build_joint_name2id_dict()
      self._build_motor_id_list()
      self._record_mass_info_from_urdf()
      self.ResetPose()
    #   if self._on_rack:
    #     self._pybullet_client.createConstraint(self.quadruped, -1, -1, -1,
    #                                            self._pybullet_client.JOINT_FIXED, [0, 0, 0],
    #                                            [0, 0, 0], [0, 0, 1])
    # else:
    #   self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, INIT_POSITION,
    #                                                         INIT_ORIENTATION)
    #   self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
    # self.ResetPose(add_constraint=False)
    else:
      self._pybullet_client.resetBasePositionAndOrientation(
          self.quadruped, init_position, init_orientation)
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0],
                                              [0, 0, 0])
      self.ResetPose()
    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors

  # def ResetPose(self, add_constraint):
  #   #del add_constraint
  #   """Reset the pose of the mdoger7.

  #   Args:
  #     add_constraint: Whether to add a constraint at the joints of two feet.
  #   """
  #   for i in range(self.num_legs):
  #     del add_constraint
  #     self._ResetPoseForLeg(i, add_constraint)

  # def _ResetPoseForLeg(self, leg_id, add_constraint):
  #   """Reset the initial pose for the leg.

  #   Args:
  #     leg_id: It should be 0, 1, 2, 3
  #     add_constraint: Whether to add a constraint at the joints of two feet.
  #   """
  #   del add_constraint
  #   hip = 0
  #   upper_leg_angle = 0    #45*math.pi / 180.0
  #   low_friction_force = 0

  #   lower_leg_angle = 0     #45*math.pi / 180.0

  #   leg_position = LEG_POSITION[leg_id]
  #   self._pybullet_client.resetJointState(self.quadruped,
  #                                         self._joint_name_to_id[leg_position +
  #                                                                str(1)+"_joint"],
  #                                         self._h1motor_direction[leg_id] * hip,
  #                                         targetVelocity=0)
  #   self._pybullet_client.resetJointState(self.quadruped,
  #                                         self._joint_name_to_id[leg_position +
  #                                                                str(2)+"_joint"],
  #                                         self._h2motor_direction[leg_id] * upper_leg_angle,
  #                                         targetVelocity=0)
  #   self._pybullet_client.resetJointState(self.quadruped,
  #                                         self._joint_name_to_id[leg_position +
  #                                                                str(3)+"_joint"],
  #                                         self._h3motor_direction[leg_id] * lower_leg_angle,
  #                                         targetVelocity=0)

  def ResetPose(self):
    # del add_constraint
    for name, i in zip(motor_names, range(len(motor_names))):
      angle = init_motor_angles[i]
      self._pybullet_client.resetJointState(self.quadruped,
                                            self._joint_name_to_id[name],
                                            angle,
                                            targetVelocity=0)
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)

    # if add_constraint:
    #   self._pybullet_client.createConstraint(
    #       self.quadruped, self._joint_name_to_id[leg_position + str(3)+"_joint"],
    #       self.quadruped, self._joint_name_to_id[leg_position + str(3)+"_joint"],
    #       self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0], lower_CONSTRAINT_POINT_RIGHT,
    #       lower_CONSTRAINT_POINT_LEFT)

    # if self._accurate_motor_model_enabled or self._pd_control_enabled:
    #   # Disable the default motor in pybullet.
    #   self._pybullet_client.setJointMotorControl2(
    #       bodyIndex=self.quadruped,
    #       jointIndex=(self._joint_name_to_id[leg_position + str(1)+"_joint"]),
    #       controlMode=self._pybullet_client.VELOCITY_CONTROL,
    #       targetVelocity=0,
    #       force=low_friction_force)

    # else:
    #   self._SetDesiredMotorAngleByName(leg_position + str(1)+"_joint",
    #                                    self._h1motor_direction[leg_id] * hip)
    # self._pybullet_client.setJointMotorControl2(
    #     bodyIndex=self.quadruped,
    #     jointIndex=(self._joint_name_to_id[leg_position + str(2)+"_joint"]),
    #     controlMode=self._pybullet_client.VELOCITY_CONTROL,
    #     targetVelocity=0,
    #     force=low_friction_force)
    # self._pybullet_client.setJointMotorControl2(
    #     bodyIndex=self.quadruped,
    #     jointIndex=(self._joint_name_to_id[leg_position + str(3)+"_joint"]),
    #     controlMode=self._pybullet_client.VELOCITY_CONTROL,
    #     targetVelocity=0,
    #     force=low_friction_force)

  def CheckJointContact(self):
    """
    @brief: 检查指定的连杆是否与地面接触。
    @return: 表示是否与地面接触。
    """
    link_names = [
        "lf1_joint", "lf2_joint", "lf3_joint", "rf1_joint", "rf2_joint",
        "rf3_joint", "lb1_joint", "lb2_joint", "lb3_joint", "rb1_joint",
        "rb2_joint", "rb3_joint"
    ]

    # 初始化接触检测结果
    collision_count = False
    # 转换为连杆 ID
    motor_joint_ids = [self._joint_name_to_id[name] for name in link_names]
    # 获取所有接触点
    contact_points = self._pybullet_client.getContactPoints(
        bodyA=self.ground_id, bodyB=self.quadruped)
    # 检查接触
    for contact in contact_points:
      link_id = contact[4]  # 接触的连杆 ID
      if link_id in motor_joint_ids:
        collision_count = True
    return collision_count

  # def CheckJointContact(self):
  #   """
  #   检查特定关节是否与地面接触。
  #   Args:
  #     ground_id (int): 地面的ID。
  #   Returns:
  #     dict: 一个字典，包含关节名称和它们是否与地面接触的信息。
  #   """
  #   contact_detected = {}
  #   # 关节名称
  #   joints_to_check = ["_foot_"]
  #   # 获取子链ID
  #   link_ids_to_check = [self._joint_name_to_id[joint] for joint in joints_to_check]
  #   # 检测接触
  #   contact_points = self._pybullet_client.getContactPoints(bodyA=self.quadruped, bodyB=self.ground_id)
  #   # 初始化字典
  #   for joint in joints_to_check:
  #       contact_detected[joint] = False
  #   joint_contact_count = 0
  #   # 分析接触点
  #   for contact in contact_points:
  #       link_id = contact[3]  # 接触的链接ID
  #       if link_id in link_ids_to_check:
  #           joint_name = joints_to_check[link_ids_to_check.index(link_id)]
  #           contact_detected[joint_name] = True
  #           joint_contact_count+=1
  #           collision_penalty = -joint_contact_count * 5
  #       print("collision_penalty:", collision_penalty)
  #       return collision_penalty

  # def CheckJointContact(self, joint_name_suffix="_3_joint"):
  #       """检查是否有任何名为'joint_name_suffix'的关节接触到了任何东西。
  #       Args:
  #         joint_name_suffix: 要检查的关节名称后缀。
  #       Returns:
  #         一个布尔值，如果有任何名为'joint_name_suffix'的关节发生接触，则为True。
  #       """
  #       contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped, bodyB=self.ground_id)
  #       # print("contacts:", contacts)
  #       joint_contact_count = 0
  #       for contact in contacts:
  #           contact_link_id = contact[3]  # 获取接触的链接ID
  #           contact_link_info = self._pybullet_client.getJointInfo(self.quadruped, contact_link_id)
  #           contact_link_name = contact_link_info[1].decode('UTF-8')
  #           if joint_name_suffix in contact_link_name:
  #               joint_contact_count += 1
  #       collision_penalty = -joint_contact_count * 5
  #       # print("collision_penalty:", collision_penalty)
  #       return collision_penalty

  def GetBasePosition(self):
    """Get the position of mdoger7's base.

    Returns:
      The position of mdoger7's base.
    """
    position, _ = (self._pybullet_client.getBasePositionAndOrientation(
        self.quadruped))
    return position

  def GetBaseOrientation(self):
    """Get the orientation of mdoger7's base, represented as quaternion.

    Returns:
      The orientation of mdoger7's base.
    """
    _, orientation = (self._pybullet_client.getBasePositionAndOrientation(
        self.quadruped))
    return orientation

  def GetBaseVelocity(self):
    """Get the velocity of mdoger7's base in the XY plane and the yaw rate.

    Returns:
      A tuple containing:
      - The XY velocity of mdoger7's base.
      - The yaw rate (rotational velocity around Z-axis) of mdoger7's base.
    """
    linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
        self.quadruped)
    xy_velocity = np.array(
        linear_velocity[:2])  # Take only the X and Y components
    yaw_rate = angular_velocity[2]  # Z-axis component represents yaw rate
    return xy_velocity, yaw_rate

  def GetActionDimension(self):
    """Get the length of the action list.

    Returns:
      The length of the action list.
    """
    return self.num_motors

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    # upper_bound = np.array([0.0] * self.GetObservationDimension())
    # upper_bound[0:self.num_motors] = math.pi  # Joint angle.
    # upper_bound[self.num_motors:2 * self.num_motors] = (motor.MOTOR_SPEED_LIMIT)  # Joint velocity.
    # upper_bound[2 * self.num_motors:3 * self.num_motors] = (motor.OBSERVED_TORQUE_LIMIT
    #                                                        )  # Joint torque.
    # upper_bound[3 * self.num_motors:] = 1.0  # Quaternion of base orientation.
    upper_bound = np.array([0.0] * self.GetObservationDimension())
    upper_bound[0:self.num_motors] = math.pi  # Joint angle.
    upper_bound[self.num_motors:2 * self.num_motors] = self.parameters_list['motor']['motor_speed_limit'] # Joint velocity.
    upper_bound[2 * self.num_motors:3 * self.num_motors] = self.parameters_list['motor']['observed_torque_limit']  # Joint torque.
    upper_bound[3 * self.num_motors:3 * self.num_motors +
                4] = 1.0  # Quaternion of base orientation.

    # Assuming a reasonable upper limit for base XY velocity (e.g., 10 m/s)
    upper_bound[3 * self.num_motors + 4:3 * self.num_motors +
                6] = 10.0  # Upper limit for XY velocity

    # Assuming a reasonable upper limit for yaw rate (e.g., 5 rad/s)
    upper_bound[3 * self.num_motors + 6] = 5.0  # Upper limit for yaw rate

    return upper_bound

  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    return -self.GetObservationUpperBound()

  def GetObservationDimension(self):
    """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
    return len(self.GetObservation())

  def GetObservation(self):
    """Get the observations of mdoger.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:12] are motor angles. observation[12:24]
      are motor velocities, observation[24:36] are motor torques.
      observation[36:48] is the orientation of the base, in quaternion form.
    """
    observation = []
    observation.extend(self.GetMotorAngles().tolist())
    observation.extend(self.GetMotorVelocities().tolist())
    observation.extend(self.GetMotorTorques().tolist())
    observation.extend(list(self.GetBaseOrientation()))
    xy_velocity, yaw_rate = self.GetBaseVelocity()
    observation.extend(
        xy_velocity.tolist())  # Add XY velocity to the observation
    observation.append(yaw_rate)
    # print('******************')
    # print('observation:', observation)# Add yaw rate to the observation
    return observation

  def ApplyAction(self, motor_commands):
    """Set the desired motor angles to the motors of the mdoger.

    The desired motor angles are clipped based on the maximum allowed velocity.
    If the pd_control_enabled is True, a torque is calculated according to
    the difference between current and desired joint angle, as well as the joint
    velocity. This torque is exerted to the motor. For more information about
    PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.

    Args:
      motor_commands: The 12 desired motor angles.
    """
    if self._motor_velocity_limit < np.inf:
      current_motor_angle = self.GetMotorAngles()
      motor_commands_max = (current_motor_angle +
                            self.time_step * self._motor_velocity_limit)
      motor_commands_min = (current_motor_angle -
                            self.time_step * self._motor_velocity_limit)
      motor_commands = np.clip(motor_commands, motor_commands_min,
                               motor_commands_max)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      q = self.GetMotorAngles()
      qdot = self.GetMotorVelocities()
      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
            motor_commands, q, qdot)
        if self._motor_overheat_protection:
          for i in range(self.num_motors):
            if abs(actual_torque[i]) > overheat_shutdown_torque:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] = 0
            if (self._overheat_counter[i]
                > overheat_shutdown_time / self.time_step):
              self._motor_enabled_list[i] = False

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque,
                                                 self._motor_direction)

        for motor_id, motor_torque, motor_enabled in zip(
            self._motor_id_list, self._applied_motor_torque,
            self._motor_enabled_list):
          if motor_enabled:
            self._set_motor_torque_by_id(motor_id, motor_torque)
          else:
            self._set_motor_torque_by_id(motor_id, 0)
      else:
        torque_commands = -self._kp * (q - motor_commands) - self._kd * qdot

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = torque_commands

        # Transform into the motor space when applying the torque.
        self._applied_motor_torques = np.multiply(self._observed_motor_torques,
                                                  self._motor_direction)

        for motor_id, motor_torque in zip(self._motor_id_list,
                                          self._applied_motor_torques):
          self._set_motor_torque_by_id(motor_id, motor_torque)
    else:
      motor_commands_with_direction = np.multiply(motor_commands,
                                                  self._motor_direction)
      for motor_id, motor_command_with_direction in zip(
          self._motor_id_list, motor_commands_with_direction):
        self._set_desired_motor_angle_by_id(motor_id, motor_command_with_direction)

  def GetMotorAngles(self):
    """Get the eight motor angles at the current moment.

    Returns:
      Motor angles.
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
        for motor_id in self._motor_id_list
    ]
    motor_angles = np.multiply(motor_angles, self._motor_direction)
    return motor_angles

  def GetMotorVelocities(self):
    """Get the velocity of all 12 motors.

    Returns:
      Velocities of all 12 motors.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
        for motor_id in self._motor_id_list
    ]
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorTorques(self):
    """Get the amount of torques the motors are exerting.

    Returns:
      Motor torques of all 12 motors.
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
          self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
          for motor_id in self._motor_id_list
      ]
      motor_torques = np.multiply(motor_torques, self._motor_direction)
    return motor_torques

  def ConvertFromLegModel(self, actions):
    motor_angle = copy.deepcopy(actions)
    for i in range(len(actions)):
      motor_angle[i] = math.fmod(actions[i], math.pi)
      if motor_angle[i] >= math.pi / 2:
        motor_angle[i] -= math.pi
      elif motor_angle[i] < -math.pi:
        motor_angle[i] += math.pi
    return motor_angle

  # def ConvertFromLegModel(self, actions):
  #   """Convert the actions that use leg model to the real motor actions.

  #   Args:
  #     actions: The theta, phi of the leg model.
  #   Returns:
  #     The eight desired motor angles that can be used in ApplyActions().
  #   """

  #   motor_angle = copy.deepcopy(actions)
  #   scale_for_singularity = 1.5
  #   offset_for_singularity = 1.5
  #   half_num_motors = int(self.num_motors / 2)
  #   quater_pi =1 * math.pi / 4
  #   for i in range(self.num_motors):
  #     action_idx = i // 2
  #     forward_backward_component = (
  #         -scale_for_singularity * quater_pi *
  #         (actions[action_idx + half_num_motors] + offset_for_singularity))
  #     extension_component = (-1)**i * quater_pi * actions[action_idx]
  #     if i >= half_num_motors:
  #       extension_component = -extension_component
  #     motor_angle[i] = (3*math.pi  / 4 + forward_backward_component + extension_component)
  #   return motor_angle

  def GetBaseMassFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def SetBaseMass(self, base_mass):
    self._pybullet_client.changeDynamics(self.quadruped,
                                         base_link_id,
                                         mass=base_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. All four leg_links have the same mass,
    which is leg_masses[0]. All four motors have the same mass, which is
    leg_mass[1].

    Args:
      leg_masses: The leg masses. leg_masses[0] is the mass of the leg link.
        leg_masses[1] is the mass of the motor.
    """
    # for link_id in LEG_LINK_ID:
    #   self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=leg_masses[0])
    for link_id in motor_link_id:
      self._pybullet_client.changeDynamics(self.quadruped,
                                    link_id,
                                    mass=leg_masses[0])

  def SetFootFriction(self, foot_friction):
    """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in foot_link_id:
      self._pybullet_client.changeDynamics(self.quadruped,
                                    link_id,
                                    lateralFriction=foot_friction)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)
