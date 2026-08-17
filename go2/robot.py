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

from go2.constants import (
  BASE_MASS,
  DEFAULT_HEIGHT,
  DEFAULT_JOINT_ANGLES,
  FEET,
  FEET_CONTACT_SENSOR,
  FOOT_GEOMS,
  FOOT_SITE_POS,
  GO2_ACTUATORS,
  GO2_MJCF_PATH,
  JOINT_NAMES,
  JOINT_REGEX,
  JOINT_TYPES,
)

# Command latency, in *physics* timesteps (sim.timestep = 0.0025 s), sampled per
# env per step. 0-2 steps = 0-5 ms = up to one full control period, modelling the
# DDS/bus lag between the policy and the motor boards on the real robot.
COMMAND_DELAY_STEPS = (0, 2)


def get_spec() -> mujoco.MjSpec:
  """Build the Go2 spec from the pinned ``go2_mjcf`` submodule.

  The MJCF is an edited copy of Menagerie's ``unitree_go2``, pinned by submodule
  SHA and shared with a separate trajectory-optimization project so the two agree
  on one robot. Its base is byte-identical to what ``robot_descriptions`` pulls --
  same topology, inertias and joint ranges (total mass 15.206408 kg) -- but it
  ships several things this file used to inject, most importantly:

    * the four ``<leg>_site`` foot sites and a 30-entry ``<sensor>`` block (which
      includes the ``accelerometer`` the privileged observation reads). Injecting
      either here now raises ``repeated name`` at compile time.
    * ``margin = 0`` on every collision geom, and flattened ``<default>`` classes
      with every joint/motor attribute inlined. The flattening is load-bearing for
      the Drake side of that other project -- Drake merges a default class with its
      *immediate parent only*, so upstream's depth-2 ``front_hip``/``back_hip``
      classes silently parsed with the wrong axis and zero armature. Do not tidy
      the XML back into nested classes.

  Only two deltas remain, both from ``GO2_ACTUATORS``:

    * joint dynamics (armature / damping / frictionloss). These have to be written
      here rather than left to ``_actuator_cfgs()``: ``XmlActuatorCfg`` accepts the
      three fields but never applies them (see ``_apply_joint_dynamics``).
    * effort limits, on both ``ctrlrange`` and ``forcerange`` - keeps MuJoCo's
      clamp in step with Drake's torque saturation and with the action scale. The
      XML carries the datasheet peaks (23.7 / 45.43); this table derates them.

  The ``<keyframe>`` block (``home`` and ``tuck``) is dropped because mjlab sets
  the initial state from ``InitialStateCfg`` instead.
  """
  spec = mujoco.MjSpec.from_file(str(GO2_MJCF_PATH))

  _apply_joint_dynamics(spec)

  for actuator in spec.actuators:
    joint_type = actuator.target.split("/")[-1].split("_")[1]
    limit = GO2_ACTUATORS[joint_type].effort_limit
    actuator.ctrlrange = [-limit, limit]
    actuator.forcerange = [-limit, limit]

  for key in list(spec.keys):
    spec.delete(key)
  return spec


def _apply_joint_dynamics(spec: mujoco.MjSpec) -> None:
  """Write ``GO2_ACTUATORS``' armature / damping / frictionloss onto the joints.

  This exists because **``XmlActuatorCfg`` silently ignores those three fields**.
  ``ActuatorCfg`` documents ``armature``/``frictionloss``/``viscous_damping`` as
  "None preserves the XML value", but only the PD and builtin actuator paths ever
  call ``mjlab.utils.spec``'s writers; ``XmlActuator.edit_spec`` just wraps the
  existing ``<motor>`` and returns (verified against mjlab 1.6.0). So under the
  flat task's torque control the compiled model takes its joint dynamics straight
  from the XML, and ``_actuator_cfgs()``'s arguments are decorative.

  That went unnoticed while the XML happened to agree with the table. It does not
  agree any more -- ``go2_mjcf`` carries damping 0.05 and no frictionloss, tuned
  for a trajectory-optimization consumer that supplies its own actuator model --
  so without this the flat task would quietly lose 40x its joint damping and all
  of its dry friction, and the ``scale`` DR terms would randomize about the wrong
  centre.

  The rugged task does not need this (``BuiltinPositionActuatorCfg`` really does
  apply the overrides) but is unharmed by it: ``rugged/robot.get_spec()`` wraps
  this one, and mjlab then rewrites the same three fields with its own values.
  """
  for joint_name in JOINT_NAMES:
    joint_type = joint_name.split("_")[1]
    group = GO2_ACTUATORS[joint_type]
    joint = spec.joint(joint_name)
    joint.armature = group.armature
    joint.frictionloss = group.frictionloss
    joint.damping[0] = group.damping


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
  control). It is *only* used here for the delay model and the target regexes:
  the ``armature`` / ``frictionloss`` / ``viscous_damping`` below are passed for
  documentation and are **not** what makes them true -- ``XmlActuator.edit_spec``
  drops them on the floor. ``get_spec()``'s ``_apply_joint_dynamics`` is what
  actually writes them, and it uses this same table, so the values agree.
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
  """Assert the ``go2_mjcf`` model still matches what this project assumes.

  The model is pinned by submodule SHA, so it can no longer drift underneath us on
  its own -- but it is *shared* with another project, so a submodule bump made for
  that project's benefit can still change the robot this one trains on. These
  assertions turn that into a loud failure. Run via ``python scripts/check_robot.py``
  after any bump.

  Note this checks ``get_spec()``'s output directly, so it sees the XML plus this
  file's deltas but *not* mjlab's actuator layer -- which for the flat task is the
  point, since ``XmlActuatorCfg`` adds nothing (see ``_apply_joint_dynamics``).
  """
  m = get_spec().compile()
  assert (m.nu, m.nq, m.nv) == (12, 19, 18), f"topology changed: {m.nu=} {m.nq=} {m.nv=}"
  assert abs(m.body_subtreemass[1] - BASE_MASS) < 1e-3, (
    f"total mass {m.body_subtreemass[1]} != BASE_MASS {BASE_MASS}"
  )
  for name in FOOT_GEOMS:
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, f"no geom {name}"
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}_site")
    assert sid >= 0, f"no site {name}_site"
    assert tuple(m.site_pos[sid]) == FOOT_SITE_POS, (
      f"{name}_site moved to {tuple(m.site_pos[sid])}, expected {FOOT_SITE_POS}"
    )
  assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer") >= 0
  assert m.geom_margin.max() == 0.0, "contact margin must be 0"

  # The four thigh joints must rotate about +y. This is the specific failure the
  # XML's flattened <default> classes exist to prevent: upstream nests
  # front_hip/back_hip two levels deep under a class that sets axis="0 1 0", and a
  # parser that merges only one level (Drake's) silently reads the MuJoCo default
  # (0, 0, 1) instead -- no warning, and a ~29x error in the thigh mass matrix.
  # MuJoCo itself merges correctly, so this only fires if someone re-nests the
  # classes *and* drops the inlined axis.
  for leg in FEET:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")
    assert jid >= 0, f"no joint {leg}_thigh_joint"
    assert tuple(m.jnt_axis[jid]) == (0.0, 1.0, 0.0), (
      f"{leg}_thigh_joint axis is {tuple(m.jnt_axis[jid])}, expected (0, 1, 0) -- "
      "has the MJCF been re-nested into default classes?"
    )

  for jt, group in GO2_ACTUATORS.items():
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"FL_{jt}")
    assert aid >= 0, f"no actuator FL_{jt}"
    assert m.actuator_ctrlrange[aid][1] == group.effort_limit, (
      f"{jt} ctrlrange {m.actuator_ctrlrange[aid]} != +/-{group.effort_limit}"
    )
    assert m.actuator_forcerange[aid][1] == group.effort_limit, (
      f"{jt} forcerange {m.actuator_forcerange[aid]} != +/-{group.effort_limit}"
    )
    # The flat task gets these from get_spec(), not from mjlab -- assert them here
    # or nothing does.
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"FL_{jt}_joint")
    dof = m.jnt_dofadr[jid]
    assert m.dof_armature[dof] == group.armature, f"{jt} armature"
    assert m.dof_damping[dof] == group.damping, f"{jt} damping"
    assert m.dof_frictionloss[dof] == group.frictionloss, f"{jt} frictionloss"
  return m
