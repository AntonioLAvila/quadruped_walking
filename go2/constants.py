"""Physical/robot constants for the Unitree Go2.

Pure data only (stdlib imports only) so this module can be imported from any
environment that needs to agree with mjlab's Go2 setup -- including
``verification.py``'s separate pydrake environment, which has neither
``mujoco`` nor ``mjlab`` installed.

mjlab-specific config builders (``EntityCfg``, ``ContactSensorCfg``, etc.)
live in ``go2_robot.py``, which imports the constants defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

##
# Model topology.
#
# The MJCF lives in the ``go2_mjcf`` submodule -- a pinned, edited copy of the
# Menagerie ``unitree_go2`` model, shared with a separate trajectory-optimization
# project so both agree on one robot. ``go2/robot.py`` loads it and applies a short
# list of deltas. Only pure data lives here.
##

# Path to the submodule's MJCF, resolved from this file so it works regardless of
# the working directory. ``pathlib`` is stdlib, so the firewall (see module
# docstring) holds.
GO2_MJCF_PATH = Path(__file__).resolve().parents[1] / "go2_mjcf" / "go2.xml"

# Foot collision geoms / sites / base body, as named in the Menagerie Go2 model.
FEET = ("FL", "FR", "RL", "RR")
FOOT_GEOMS = FEET
FOOT_SITES = tuple(f"{f}_site" for f in FEET)
BASE_BODY = "base"

# Foot site offset within its parent ``<leg>_calf`` body (the foot sphere centre).
# The sites now ship in the MJCF; this stays as the documented offset and as the
# value ``check_spec()`` asserts the XML still uses.
FOOT_SITE_POS = (0.0, 0.0, -0.213)

# Default standing height (home keyframe qpos z) and base subtree mass.
# DEFAULT_HEIGHT feeds the "healthy" reward; BASE_MASS feeds the kick force scaling.
# BASE_MASS is body_subtreemass[base] from the compiled model (== total robot mass);
# verified by compiling the stripped spec.
DEFAULT_HEIGHT = 0.27
BASE_MASS = 15.206
FEET_MIN_HEIGHT = 0.01
FEET_MAX_HEIGHT = 0.15

# Name of the (mjlab-added) feet contact sensor used by the feet rewards/observations.
FEET_CONTACT_SENSOR = "feet_ground_contact"

##
# Joints: single source of truth for names, default pose, and torque limits.
##

# Per-leg joint types, in the order each leg's joints appear in the MJCF/URDF.
JOINT_TYPES = ("hip", "thigh", "calf")

# Home keyframe pose: per-leg [hip=0, thigh=0.9, calf=-1.8].
DEFAULT_JOINT_ANGLES = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}

# Explicit per-joint names in FL, FR, RL, RR x hip, thigh, calf order, e.g. for
# code (Drake) that needs to align a flat per-joint array to the model's joints.
JOINT_NAMES = tuple(f"{leg}_{jt}_joint" for leg in FEET for jt in JOINT_TYPES)

# Regex matching every joint of one type, for the mjlab configs.
JOINT_REGEX = {jt: f".*_{jt}_joint" for jt in JOINT_TYPES}

##
# Actuators: the single source of truth for per-joint actuator dynamics.
#
# Both simulators configure themselves from this table, so they cannot drift
# apart: ``go2/robot.py::get_spec()`` writes these onto the MJCF's joints at
# spec-build time, and ``scripts/verify_*.py`` feed the same numbers to Drake's
# reflected inertia, joint damping and torque saturation. Never hard-code any of
# these at a call site.
#
# Note it is ``get_spec()`` that applies them, not mjlab. ``XmlActuatorCfg``
# accepts ``armature``/``frictionloss``/``viscous_damping`` but silently ignores
# them (verified against mjlab 1.6.0: only the PD/builtin actuator paths call
# ``utils.spec``'s joint-dynamics writers). The rugged task's
# ``BuiltinPositionActuatorCfg`` really does override; the flat task's does not,
# so the flat task would otherwise inherit whatever the XML happens to ship.
#
# Values follow Unitree's own mjlab RL config (unitree_rl_mjlab,
# ``src/assets/robots/unitree_go2/go2_constants.py``) rather than Menagerie's
# generic defaults. Two deliberate differences from Menagerie:
#   * the knee's armature is 0.02, twice what Menagerie ships for every joint;
#   * effort limits are derated from the datasheet peaks (23.7 / 45.43).
##


@dataclass(frozen=True)
class ActuatorGroup:
  """Actuator parameters shared by all four instances of one joint type."""

  effort_limit: float
  """N*m. MuJoCo ``<motor ctrlrange>``, Drake's torque saturation, and -- because
  the action is a direct effort in [-1, 1] -- the action scale."""
  armature: float
  """kg*m^2. Reflected rotor inertia: MuJoCo ``<joint armature>``, and exactly
  Drake's ``JointActuator`` rotor inertia (gear ratio is 1)."""
  damping: float
  """N*m*s/rad. Passive viscous damping."""
  frictionloss: float
  """N*m. Dry (load-independent) joint friction. MuJoCo only -- Drake's
  MultibodyPlant has no equivalent, so the Drake model runs slightly looser."""
  kp: float
  """PD position gain. Unused by the torque policy, but this is what a
  position-control variant and the real robot's motor boards use."""
  kd: float
  """PD derivative gain. See ``kp``."""


GO2_ACTUATORS: dict[str, ActuatorGroup] = {
  "hip": ActuatorGroup(
    effort_limit=23.5, armature=0.01, damping=2.0, frictionloss=0.2, kp=20.0, kd=1.0
  ),
  "thigh": ActuatorGroup(
    effort_limit=23.5, armature=0.01, damping=2.0, frictionloss=0.2, kp=20.0, kd=1.0
  ),
  "calf": ActuatorGroup(
    effort_limit=45.0, armature=0.02, damping=2.0, frictionloss=0.2, kp=40.0, kd=2.0
  ),
}

# Flat per-joint arrays aligned to JOINT_NAMES, for code that indexes by joint.
DEFAULT_JOINT_POS = tuple(DEFAULT_JOINT_ANGLES[jt] for _ in FEET for jt in JOINT_TYPES)
JOINT_TORQUE_LIMITS_FLAT = tuple(
  GO2_ACTUATORS[jt].effort_limit for _ in FEET for jt in JOINT_TYPES
)
JOINT_ARMATURE_FLAT = tuple(
  GO2_ACTUATORS[jt].armature for _ in FEET for jt in JOINT_TYPES
)
JOINT_DAMPING_FLAT = tuple(GO2_ACTUATORS[jt].damping for _ in FEET for jt in JOINT_TYPES)

# Regex-keyed action scale for mjlab's JointEffortActionCfg: action ~[-1, 1] maps
# to +/- the effort limit, so the scale *is* the effort limit.
GO2_ACTION_SCALE = {JOINT_REGEX[jt]: GO2_ACTUATORS[jt].effort_limit for jt in JOINT_TYPES}

##
# Control timing.
##

SIM_TIMESTEP = 0.0025
DECIMATION = 2
CTRL_DT = SIM_TIMESTEP * DECIMATION
