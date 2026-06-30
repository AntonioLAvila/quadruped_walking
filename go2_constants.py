"""Physical/robot constants for the Unitree Go2.

Pure data only (stdlib imports only) so this module can be imported from any
environment that needs to agree with mjlab's Go2 setup -- including
``verification.py``'s separate pydrake environment, which has neither
``mujoco`` nor ``mjlab`` installed.

mjlab-specific config builders (``EntityCfg``, ``ContactSensorCfg``, etc.)
live in ``go2_robot.py``, which imports the constants defined here.
"""

from __future__ import annotations

from pathlib import Path

##
# MJCF and assets.
##

GO2_XML = Path(__file__).parent / "mjcf_go2" / "go2_warp.xml"
assert GO2_XML.exists(), f"Go2 XML not found at {GO2_XML}"

# Foot collision geoms / sites / base body, as named in go2_warp.xml.
FEET = ("FL", "FR", "RL", "RR")
FOOT_GEOMS = FEET
FOOT_SITES = tuple(f"{f}_site" for f in FEET)
BASE_BODY = "base"

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

# Per-joint-type torque limit (N*m); action (~[-1, 1]) -> torque scale.
JOINT_TORQUE_LIMITS = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}

# Explicit per-joint names in FL, FR, RL, RR x hip, thigh, calf order, e.g. for
# code (Drake) that needs to align a flat per-joint array to the model's joints.
JOINT_NAMES = tuple(f"{leg}_{jt}_joint" for leg in FEET for jt in JOINT_TYPES)

# Flat per-joint arrays aligned to JOINT_NAMES.
DEFAULT_JOINT_POS = tuple(DEFAULT_JOINT_ANGLES[jt] for _ in FEET for jt in JOINT_TYPES)
JOINT_TORQUE_LIMITS_FLAT = tuple(JOINT_TORQUE_LIMITS[jt] for _ in FEET for jt in JOINT_TYPES)

# Regex-keyed action scale for mjlab's JointEffortActionCfg.
GO2_ACTION_SCALE = {f".*_{jt}_joint": JOINT_TORQUE_LIMITS[jt] for jt in JOINT_TYPES}

##
# Control timing.
##

SIM_TIMESTEP = 0.0025
DECIMATION = 2
CTRL_DT = SIM_TIMESTEP * DECIMATION
