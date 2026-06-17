from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

##
# MJCF and assets.
##

GO2_XML: Path = Path(__file__).parent / "mjcf_go2" / "go2_warp.xml"
assert GO2_XML.exists(), f"Go2 XML not found at {GO2_XML}"

# Foot collision geoms / sites / base body, as named in go2_warp.xml.
FEET: tuple[str, ...] = ("FL", "FR", "RL", "RR")
FOOT_GEOMS: tuple[str, ...] = FEET
FOOT_SITES: tuple[str, ...] = tuple(f"{f}_site" for f in FEET)
BASE_BODY: str = "base"

# Default standing height (home keyframe qpos z) and base subtree mass.
# DEFAULT_HEIGHT feeds the "healthy" reward; BASE_MASS feeds the kick force scaling.
# BASE_MASS is body_subtreemass[base] from the compiled model (== total robot mass);
# verified by compiling the stripped spec.
DEFAULT_HEIGHT: float = 0.27
BASE_MASS: float = 15.206
FEET_MIN_HEIGHT: float = 0.01

# Name of the (mjlab-added) feet contact sensor used by the feet rewards/observations.
FEET_CONTACT_SENSOR: str = "feet_ground_contact"


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
# Actuators (torque / motor control).
##

# Per-joint action scale for JointEffortActionCfg: action (~[-1, 1]) -> torque.
GO2_ACTION_SCALE: dict[str, float] = {
  ".*_hip_joint": 23.7,
  ".*_thigh_joint": 23.7,
  ".*_calf_joint": 45.43,
}

##
# Initial state (home keyframe: z=0.27, per-leg [hip=0, thigh=0.9, calf=-1.8]).
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, DEFAULT_HEIGHT),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
  },
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
