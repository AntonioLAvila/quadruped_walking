"""Unitree Go2 robot configuration for mjlab.

Faithful port of the robot setup from ``env.py`` (MuJoCo Playground) to mjlab's
entity/actuator system. Uses **torque (motor) control** to match the original env,
which applied ``action * [23.7, 23.7, 45.43] * 4`` directly as motor ctrl.

The repo's menagerie XML (``unitree_go2/go2_warp.xml``) is incompatible with mjlab as
shipped, so ``get_spec`` performs minimal surgery (see below). Modelled on
``mjlab/asset_zoo/robots/unitree_go1/go1_constants.py``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

##
# MJCF and assets.
##

GO2_XML: Path = Path(__file__).parent / "unitree_go2" / "go2_warp.xml"
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

# Name of the (mjlab-added) feet contact sensor used by the feet rewards/observations.
FEET_CONTACT_SENSOR: str = "feet_ground_contact"


def get_spec() -> mujoco.MjSpec:
  """Load the Go2 MJCF and make it mjlab-compatible.

  Surgery (the menagerie XML ships things mjlab provides itself):
    * delete the 12 ``<motor>`` actuators  - mjlab injects its own actuators.
    * delete the ``<keyframe>``            - mjlab sets initial state from InitialStateCfg.
    * delete every ``<sensor>`` except ``accelerometer`` - the 4 ``*_floor_contact``
      sensors reference a ``floor`` geom that lives only in the scene wrapper (so a
      standalone compile would fail), and every other quantity we need is available
      via EntityData. We keep ``accelerometer`` because the privileged observation
      reads it and there is no equivalent EntityData accessor.
  """
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for key in list(spec.keys):
    spec.delete(key)
  for sensor in list(spec.sensors):
    if sensor.name != "accelerometer":
      spec.delete(sensor)
  return spec


##
# Actuators (torque / motor control).
##

# BuiltinMotorActuatorCfg creates a <motor> per joint whose compute() returns the
# effort target; effort_limit clamps it. Combined with JointEffortActionCfg this
# reproduces env.py: ctrl = clip(action * scale, +-effort_limit) applied as torque.
# Effort limits match the menagerie motor ctrlranges (+-23.7 abduction/hip, +-45.43 knee).
GO2_HIP_ACTUATOR_CFG = BuiltinMotorActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  effort_limit=23.7,
)
GO2_KNEE_ACTUATOR_CFG = BuiltinMotorActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  effort_limit=45.43,
)

# Per-joint action scale for JointEffortActionCfg: action (~[-1, 1]) -> torque.
# Reproduces env.py's action_scale = [23.7, 23.7, 45.43] * 4 (abduction, hip, knee).
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
  actuators=(GO2_HIP_ACTUATOR_CFG, GO2_KNEE_ACTUATOR_CFG),
  # Matches env.py's soft_joint_limit_factor; used for reset-clamping. The
  # dof_pos_limits reward uses its own range*0.95 soft limits (see go2_mdp.py).
  soft_joint_pos_limit_factor=0.95,
)


def get_go2_robot_cfg() -> EntityCfg:
  """Fresh Go2 EntityCfg instance (avoids shared-mutation issues)."""
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
  # Sanity check: the stripped spec compiles standalone.
  m = get_spec().compile()
  print(f"Go2 spec OK: nu={m.nu} nq={m.nq} nv={m.nv} nbody={m.nbody} nsensor={m.nsensor}")
