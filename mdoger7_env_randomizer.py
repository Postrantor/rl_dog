"""Randomize the mdoger7_gym_env when reset() is called."""
import random
import numpy as np
from pybullet_envs.bullet import env_randomizer_base

# Relative range.
mdoger7_BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
mdoger7_LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
# Absolute range.
BATTERY_VOLTAGE_RANGE = (14.8, 16.8)  # Unit: Volt
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.01)  # Unit: N*m*s/rad (torque/angular vel)
mdoger7_LEG_FRICTION = (0.8, 1.5)  # Unit: dimensionless


class mdoger7EnvRandomizer(env_randomizer_base.EnvRandomizerBase):
  """A randomizer that change the mdoger7_gym_env during every reset."""

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
    """Randomize various physical properties of mdoger7.

    It randomizes the mass/inertia of the base, mass/inertia of the legs,
    friction coefficient of the feet, the battery voltage and the motor damping
    at each reset() of the environment.

    Args:
      mdoger7: the mdoger7 instance in mdoger7_gym_env environment.
    """
    base_mass = mdoger7.GetBaseMassFromURDF()
    randomized_base_mass = random.uniform(
        base_mass * (1.0 + self._mdoger7_base_mass_err_range[0]),
        base_mass * (1.0 + self._mdoger7_base_mass_err_range[1]))
    mdoger7.SetBaseMass(randomized_base_mass)

    leg_masses = mdoger7.GetLegMassesFromURDF()
    leg_masses_lower_bound = np.array(leg_masses) * (1.0 + self._mdoger7_leg_mass_err_range[0])
    leg_masses_upper_bound = np.array(leg_masses) * (1.0 + self._mdoger7_leg_mass_err_range[1])
    randomized_leg_masses = [
        np.random.uniform(leg_masses_lower_bound[i], leg_masses_upper_bound[i])
        for i in range(len(leg_masses))
    ]
    mdoger7.SetLegMasses(randomized_leg_masses)

    randomized_battery_voltage = random.uniform(BATTERY_VOLTAGE_RANGE[0], BATTERY_VOLTAGE_RANGE[1])
    mdoger7.SetBatteryVoltage(randomized_battery_voltage)

    randomized_motor_damping = random.uniform(MOTOR_VISCOUS_DAMPING_RANGE[0],
                                              MOTOR_VISCOUS_DAMPING_RANGE[1])
    mdoger7.SetMotorViscousDamping(randomized_motor_damping)

    randomized_foot_friction = random.uniform(mdoger7_LEG_FRICTION[0], mdoger7_LEG_FRICTION[1])
    mdoger7.SetFootFriction(randomized_foot_friction)
