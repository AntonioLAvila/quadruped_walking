"""mjlab-specific Go2 scene/entity builders.

Split out of ``go2_constants.py`` because these need ``mujoco``/``mjlab``,
which aren't available in every environment that needs the plain robot
constants (e.g. ``verification.py``'s pydrake environment).
"""

from __future__ import annotations

import mujoco
from robot_descriptions import go2_mj_description

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from go2.constants import (
  BASE_MASS,
  DEFAULT_HEIGHT,
  DEFAULT_JOINT_ANGLES,
  FEET,
  FEET_CONTACT_SENSOR,
  FOOT_GEOMS,
  FOOT_SITE_POS,
  GO2_ACTUATORS,
  JOINT_REGEX,
  JOINT_TYPES,
)

# Command latency, in *physics* timesteps (sim.timestep = 0.0025 s), sampled per
# env per step. 0-2 steps = 0-5 ms = up to one full control period, modelling the
# DDS/bus lag between the policy and the motor boards on the real robot.
COMMAND_DELAY_STEPS = (0, 2)


def get_spec() -> mujoco.MjSpec:
  """Build the Go2 spec from the upstream Menagerie MJCF.

  The model is pulled from ``robot_descriptions`` rather than vendored, so it
  tracks upstream. Everything this project needs on top of it is applied here --
  the four deltas below are the entire difference, and the actuator dynamics
  (armature / damping / frictionloss) are overridden separately by
  ``GO2_ARTICULATION`` from ``GO2_ACTUATORS``, so the XML's values never matter.

    * ``margin = 0`` on every geom  - upstream ships 0.001, which inflates contact
      detection distance and makes the feet appear to touch down early.
    * add the four ``<leg>_site`` foot sites - consumed by the feet rewards and the
      privileged foot observations. The ``imu`` site is already upstream.
    * add the ``accelerometer`` sensor - the privileged observation reads it and
      there is no equivalent EntityData accessor. Upstream ships no sensors.
    * effort limits from ``GO2_ACTUATORS`` - keeps MuJoCo's ctrl clamp in step with
      Drake's torque saturation and with the action scale.

  The ``<keyframe>`` is dropped because mjlab sets the initial state from
  ``InitialStateCfg`` instead.
  """
  spec = mujoco.MjSpec.from_file(str(go2_mj_description.MJCF_PATH))

  for geom in spec.geoms:
    geom.margin = 0.0

  for leg in FEET:
    spec.body(f"{leg}_calf").add_site(
      name=f"{leg}_site", pos=FOOT_SITE_POS, size=[0.01, 0.0, 0.0], group=4
    )

  accel = spec.add_sensor()
  accel.name = "accelerometer"
  accel.type = mujoco.mjtSensor.mjSENS_ACCELEROMETER
  accel.objtype = mujoco.mjtObj.mjOBJ_SITE
  accel.objname = "imu"

  for actuator in spec.actuators:
    joint_type = actuator.target.split("/")[-1].split("_")[1]
    limit = GO2_ACTUATORS[joint_type].effort_limit
    actuator.ctrlrange = [-limit, limit]

  for key in list(spec.keys):
    spec.delete(key)
  return spec


##
# Initial state (home keyframe: z=0.27, per-leg [hip=0, thigh=0.9, calf=-1.8]).
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, DEFAULT_HEIGHT),
  joint_pos={f".*_{jt}_joint": DEFAULT_JOINT_ANGLES[jt] for jt in JOINT_TYPES},
  joint_vel={".*": 0.0},
)

def _actuator_cfgs(command_delay: bool) -> tuple[XmlActuatorCfg, ...]:
  """One actuator group per joint type, parameterised from ``GO2_ACTUATORS``.

  ``XmlActuatorCfg`` keeps the MJCF's ``<motor>`` transmission (i.e. torque
  control), but mjlab treats a non-``None`` ``armature`` / ``frictionloss`` /
  ``viscous_damping`` as an override of the XML, so this table wins over whatever
  the upstream model happens to ship. That makes ``go2_constants.py`` the single
  source of truth rather than the MJCF.
  """
  delay_min, delay_max = COMMAND_DELAY_STEPS if command_delay else (0, 0)
  return tuple(
    XmlActuatorCfg(
      target_names_expr=(JOINT_REGEX[joint_type],),
      armature=group.armature,
      frictionloss=group.frictionloss,
      viscous_damping=group.damping,
      delay_min_lag=delay_min,
      delay_max_lag=delay_max,
    )
    for joint_type, group in GO2_ACTUATORS.items()
  )


def get_go2_robot_cfg(command_delay: bool = True) -> EntityCfg:
  """Build the Go2 entity config.

  Args:
    command_delay: randomize actuator command latency. Disable for deterministic
      evaluation (``play=True``), the same way the other DR terms are disabled.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=_actuator_cfgs(command_delay),
      soft_joint_pos_limit_factor=0.95,
    ),
  )


def get_feet_contact_sensor_cfg() -> ContactSensorCfg:
  """Contact sensor for the four feet vs. the flat terrain plane.

  Provides per-foot contact (``found``), net force, and air-time tracking used by
  the feet rewards and the privileged contact/air-time observations. The terrain
  plane body is named ``terrain`` by mjlab's TerrainEntity.
  """
  return ContactSensorCfg(
    name=FEET_CONTACT_SENSOR,
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


def check_spec() -> mujoco.MjModel:
  """Assert the upstream model still matches what this project assumes.

  The MJCF is no longer vendored, so an upstream Menagerie change could silently
  alter inertias or topology underneath us. These assertions turn that into a
  loud failure. Run via ``python scripts/check_robot.py``.
  """
  m = get_spec().compile()
  assert (m.nu, m.nq, m.nv) == (12, 19, 18), f"topology changed: {m.nu=} {m.nq=} {m.nv=}"
  assert abs(m.body_subtreemass[1] - BASE_MASS) < 1e-3, (
    f"total mass {m.body_subtreemass[1]} != BASE_MASS {BASE_MASS}"
  )
  for name in FOOT_GEOMS:
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, f"no geom {name}"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}_site") >= 0
  assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer") >= 0
  assert m.geom_margin.max() == 0.0, "contact margin must be 0"
  for jt, group in GO2_ACTUATORS.items():
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"FL_{jt}")
    assert aid >= 0, f"no actuator FL_{jt}"
    assert m.actuator_ctrlrange[aid][1] == group.effort_limit, (
      f"{jt} ctrlrange {m.actuator_ctrlrange[aid]} != +/-{group.effort_limit}"
    )
  return m
