"""mjlab-specific Go2 scene/entity builders.

Split out of ``go2_constants.py`` because these need ``mujoco``/``mjlab``,
which aren't available in every environment that needs the plain robot
constants (e.g. ``verification.py``'s pydrake environment).
"""

from __future__ import annotations

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from go2_constants import (
  DEFAULT_HEIGHT,
  DEFAULT_JOINT_ANGLES,
  FEET_CONTACT_SENSOR,
  FOOT_GEOMS,
  GO2_XML,
  JOINT_TYPES,
)


def get_spec() -> mujoco.MjSpec:
  """Load the Go2 MJCF and make it mjlab-compatible.
    * delete the ``<keyframe>``            - mjlab sets initial state from InitialStateCfg.
    * delete every ``<sensor>`` except ``accelerometer`` - the 4 ``*_floor_contact``
      sensors reference a ``floor`` geom that lives only in the scene wrapper (so a
      standalone compile would fail), and every other quantity we need is available
      via EntityData. We keep ``accelerometer`` because the privileged observation
      reads it and there is no equivalent EntityData accessor.
  """
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  for key in list(spec.keys):
    spec.delete(key)
  for sensor in list(spec.sensors):
    if sensor.name != "accelerometer":
      spec.delete(sensor)
  return spec


##
# Initial state (home keyframe: z=0.27, per-leg [hip=0, thigh=0.9, calf=-1.8]).
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, DEFAULT_HEIGHT),
  joint_pos={f".*_{jt}_joint": DEFAULT_JOINT_ANGLES[jt] for jt in JOINT_TYPES},
  joint_vel={".*": 0.0},
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=('.*',)),),
  soft_joint_pos_limit_factor=0.95,
)


def get_go2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
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


if __name__ == "__main__":
  m = get_spec().compile()
  print(f"Go2 spec OK: nu={m.nu} nq={m.nq} nv={m.nv} nbody={m.nbody} nsensor={m.nsensor}")
