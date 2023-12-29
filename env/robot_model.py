# -*- coding: utf-8 -*-
"""This file implements the functionalities of a minitaur using pybullet.
"""

import copy
import math
import numpy as np
import os
import pybullet_data

# from self
from env import motor

INIT_POSITION = [0, 0, 0.3]
INIT_ORIENTATION = [0, 0, 0, 1]
lower_CONSTRAINT_POINT_RIGHT = [0, 0.00, 0.]
lower_CONSTRAINT_POINT_LEFT = [0, 0.0, 0.]
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
LEG_POSITION = ["lf", "rf", "lb", "rb"]
MOTOR_NAMES = [
    "lf1_joint", "lf2_joint", "lf3_joint", "rf1_joint", "rf2_joint",
    "rf3_joint", "lb1_joint", "lb2_joint", "lb3_joint", "rb1_joint",
    "rb2_joint", "rb3_joint"
]

NUM_LEGS = 4
DEFAULT_ABDUCTION_ANGLE = 0.0
DEFAULT_HIP_ANGLE = 0
DEFAULT_KNEE_ANGLE = 0
# Bases on the readings from 's default pose.
INIT_MOTOR_ANGLES = [
    DEFAULT_ABDUCTION_ANGLE, DEFAULT_HIP_ANGLE, DEFAULT_KNEE_ANGLE
] * NUM_LEGS
# INIT_MOTOR_ANGLES = 0*[1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]

#LEG_LINK_ID = [2, 3, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 21, 22, 24, 25]
#MOTOR_LINK_ID = [1, 4, 7, 10, 14, 17, 20, 23]
#FOOT_LINK_ID = [3, 6, 9, 12, 16, 19, 22, 25]
##BASE_LINK_ID = -1
# LEG_LINK_ID = []
MOTOR_LINK_ID = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
FOOT_LINK_ID = [3, 7, 11, 115]
BASE_LINK_ID = -1


class mdoger7():
  """
  The mdoger7 class that simulates a quadruped robot from Ghost Robotics.
  """

  def __init__(self,
               pybullet_client,
               urdf_root=os.path.join(os.path.dirname(__file__)),
               time_step=0.01,
               self_collision_enabled=False,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               accurate_motor_model_enabled=False,
               motor_kp=8.0,
               motor_kd=0.2,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               on_rack=False,
               kd_for_pd_controllers=0.3):
    """Constructs a mdoger7 and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
      urdf_root: The path to the urdf folder.
      time_step: The time step of the simulation.
      self_collision_enabled: Whether to enable self collision.
      motor_velocity_limit: The upper limit of the motor velocity.
      pd_control_enabled: Whether to use PD control for the motors.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_kp: proportional gain for the accurate motor model
      motor_kd: derivative gain for the acurate motor model
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in mdoger7.py for more
        details.
      on_rack: Whether to place the mdoger7 on rack. This is only used to debug
        the walking gait. In this mode, the mdoger7's base is hanged midair so
        that its walking gait is clearer to visualize.
      kd_for_pd_controllers: kd value for the pd controllers of the motors.
    """
    self.num_motors = 12
    self.num_legs = int(self.num_motors / 3)
    self._pybullet_client = pybullet_client
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._pd_control_enabled = pd_control_enabled
    self._motor_direction = [-1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1]
    self._h1motor_direction = [-1, -1, 1, 1]
    self._h2motor_direction = [-1, 1, 1, -1]
    self._h3motor_direction = [1, -1, -1, 1]
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 15
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self.ground_id = self._pybullet_client.loadURDF(
        "%s/plane.urdf" % pybullet_data.getDataPath())
    if self._accurate_motor_model_enabled:
      self._kp = motor_kp
      self._kd = motor_kd
      self._motor_model = motor.MotorModel(
          torque_control_enabled=self._torque_control_enabled,
          kp=self._kp,
          kd=self._kd)
    elif self._pd_control_enabled:
      self._kp = 1
      self._kd = kd_for_pd_controllers
    else:
      self._kp = 1
      self._kd = 1
    self.time_step = time_step
    self.Reset()

  def _RecordMassInfoFromURDF(self):
    self._base_mass_urdf = self._pybullet_client.getDynamicsInfo(
        self.quadruped, BASE_LINK_ID)[0]
    self._leg_masses_urdf = []
    # self._leg_masses_urdf.append(
    #     self._pybullet_client.getDynamicsInfo(self.quadruped, LEG_LINK_ID[0])[0])
    self._leg_masses_urdf.append(
        self._pybullet_client.getDynamicsInfo(self.quadruped,
                                              MOTOR_LINK_ID[0])[0])

  def _BuildJointNameToIdDict(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _BuildMotorIdList(self):
    self._motor_id_list = [
        self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES
    ]

  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.POSITION_CONTROL,
        targetPosition=desired_angle,
        positionGain=self._kp,
        velocityGain=self._kd,
        force=self._max_force)

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name],
                                   desired_angle)

  def Reset(self, reload_urdf=True):
    """
    @brief: reset the mdoger7 to its initial states.
    @param: reload_urdf: whether to reload the urdf file.
    """

    if reload_urdf:
      if self._self_collision_enabled:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/mdoger7.urdf" % self._urdf_root,
            INIT_POSITION,
            INIT_ORIENTATION,
            useFixedBase=self._on_rack,
            flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
      else:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/mdoger7.urdf" % self._urdf_root,
            INIT_POSITION,
            INIT_ORIENTATION,
            useFixedBase=self._on_rack)
      self._BuildJointNameToIdDict()
      self._BuildMotorIdList()
      self._RecordMassInfoFromURDF()
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
          self.quadruped, INIT_POSITION, INIT_ORIENTATION)
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
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      angle = INIT_MOTOR_ANGLES[i]
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
    upper_bound[self.num_motors:2 *
                self.num_motors] = motor.MOTOR_SPEED_LIMIT  # Joint velocity.
    upper_bound[2 * self.num_motors:3 *
                self.num_motors] = motor.OBSERVED_TORQUE_LIMIT  # Joint torque.
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
            if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] = 0
            if (self._overheat_counter[i]
                > OVERHEAT_SHUTDOWN_TIME / self.time_step):
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
            self._SetMotorTorqueById(motor_id, motor_torque)
          else:
            self._SetMotorTorqueById(motor_id, 0)
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
          self._SetMotorTorqueById(motor_id, motor_torque)
    else:
      motor_commands_with_direction = np.multiply(motor_commands,
                                                  self._motor_direction)
      for motor_id, motor_command_with_direction in zip(
          self._motor_id_list, motor_commands_with_direction):
        self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction)

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
                                         BASE_LINK_ID,
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
    for link_id in MOTOR_LINK_ID:
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           mass=leg_masses[0])

  def SetFootFriction(self, foot_friction):
    """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in FOOT_LINK_ID:
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           lateralFriction=foot_friction)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)
