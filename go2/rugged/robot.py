"""Robot spec, PD actuators and terrain sensors for the rugged-terrain task.

Wraps ``go2.robot`` rather than duplicating it: ``get_spec()`` below calls
``go2.robot.get_spec()`` and applies one further delta. Nothing in ``go2/`` is modified,
so the flat task cannot regress.
"""

from __future__ import annotations

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)

from go2 import robot as flat_robot
from go2.constants import BASE_MASS, FEET, FOOT_GEOMS, GO2_ACTUATORS, JOINT_REGEX
from go2.rugged.constants import (
  BASE_HEIGHT_SENSOR,
  COMMAND_DELAY_S,
  FOOT_HEIGHT_SENSOR,
  POSITION_TIMING,
  POSITION_VISCOUS_DAMPING,
  SHANK_CONTACT_SENSOR,
  TERRAIN_SCAN_SENSOR,
  THIGH_CONTACT_SENSOR,
  TRUNK_CONTACT_SENSOR,
)

##
# Non-foot collision geom names, as the ``go2_mjcf`` MJCF names them.
#
# These used to be generated here: upstream Menagerie names only the four foot spheres,
# so this file walked the spec and named the other 19 group-3 geoms
# ``<body>_collision<i>``. ``go2_mjcf`` names all 23 itself, and its names are *not* the
# generated ones -- so the naming pass is gone and the sensor patterns below point at the
# XML's names instead. The counts are unchanged (3 trunk, 1 per thigh, 2 per calf), so
# the sensors see exactly the same geometry as before.
#
# Matching by *body* is still not a workaround for any of these: the foot geom is a child
# of the calf body, so a calf-body match would fire on every footstep.
##

TRUNK_COLLISION_GEOMS = ("torso_box", "head_cyl", "head_sphere")
THIGH_COLLISION_GEOMS = tuple(f"{leg}_thigh_col" for leg in FEET)
SHANK_COLLISION_GEOMS = tuple(
  f"{leg}_calf_{part}" for leg in FEET for part in ("upper", "lower")
)


def get_spec() -> mujoco.MjSpec:
  """Flat-task spec plus the one delta PD control needs.

  **Delete the ``<actuator>`` block.** ``BuiltinPositionActuatorCfg`` *adds*
  ``<position>`` elements and does not remove the 12 ``<motor>`` elements the MJCF
  ships. Leaving them yields nu=24: ``Entity._add_initial_state_keyframe`` writes
  ctrl for every actuator in the spec, so the orphaned motors would apply a constant
  torque equal to the default joint angle, and mjlab's ONNX metadata exporter keys its
  joint->ctrl map on the actuator target, so it would silently record the wrong element.
  (mjlab's own go1.xml ships no ``<actuator>`` block at all, which is why this never
  bites the reference task.)

  The collision-geom naming pass that used to live here is gone; see the geom-name
  constants above.
  """
  spec = flat_robot.get_spec()

  for actuator in list(spec.actuators):
    spec.delete(actuator)

  return spec


def _actuator_cfgs(command_delay: bool) -> tuple[BuiltinPositionActuatorCfg, ...]:
  """One PD position actuator group per joint type, from ``GO2_ACTUATORS``.

  ``stiffness``/``damping`` are Unitree's own Go2 gains, carried unused in the table until
  now. ``viscous_damping`` deliberately does NOT use the table's ``damping`` -- see
  ``POSITION_VISCOUS_DAMPING``.
  """
  lo, hi = COMMAND_DELAY_S if command_delay else (0.0, 0.0)
  delay_min = round(lo / POSITION_TIMING.sim_timestep)
  delay_max = round(hi / POSITION_TIMING.sim_timestep)
  return tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(JOINT_REGEX[joint_type],),
      stiffness=group.kp,
      damping=group.kd,
      effort_limit=group.effort_limit,
      armature=group.armature,
      frictionloss=group.frictionloss,
      viscous_damping=POSITION_VISCOUS_DAMPING[joint_type],
      delay_min_lag=delay_min,
      delay_max_lag=delay_max,
    )
    for joint_type, group in GO2_ACTUATORS.items()
  )


def get_go2_robot_cfg(command_delay: bool = True) -> EntityCfg:
  """Rugged Go2 entity: same body, PD position actuators instead of torque."""
  return EntityCfg(
    init_state=flat_robot.INIT_STATE,
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=_actuator_cfgs(command_delay),
      soft_joint_pos_limit_factor=0.95,
    ),
  )


##
# Sensors.
#
# ``include_geom_groups=(0,)`` restricts rays to the terrain: the generator's geoms take
# MuJoCo's default group 0, while the robot's collision geoms are group 3, visuals group 2
# and the foot sites group 4.
##


def get_feet_contact_sensor_cfg() -> ContactSensorCfg:
  """The flat task's foot contact sensor, plus per-substep force history."""
  cfg = flat_robot.get_feet_contact_sensor_cfg()
  cfg.history_length = POSITION_TIMING.decimation
  return cfg


def get_terrain_scan_sensor_cfg() -> RayCastSensorCfg:
  """Height map around the base. Critic-only -- the actor never sees this.

  ``ray_alignment="yaw"`` keeps the grid level with the world as the body pitches and
  rolls, so the samples describe the ground rather than the robot's attitude.
  """
  return RayCastSensorCfg(
    name=TERRAIN_SCAN_SENSOR,
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),  # 17 x 11 = 187 rays
    ray_alignment="yaw",
    max_distance=5.0,
    include_geom_groups=(0,),
  )


def get_foot_height_sensor_cfg() -> TerrainHeightSensorCfg:
  """Per-foot clearance above local terrain, replacing absolute world z.

  ``reduction="min"`` takes the closest ground point under each foot. Note the foot site
  sits at the foot sphere's *centre* (radius 0.022), so a foot resting on flat ground reads
  ~0.022 -- numerically the same as the world z the flat task's foot rewards use, which is
  why ``FEET_MAX_HEIGHT`` carries over unchanged.
  """
  return TerrainHeightSensorCfg(
    name=FOOT_HEIGHT_SENSOR,
    frame=tuple(ObjRef(type="site", name=f"{leg}_site", entity="robot") for leg in FEET),
    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
    reduction="min",
    max_distance=1.0,
    include_geom_groups=(0,),
  )


def get_base_height_sensor_cfg() -> TerrainHeightSensorCfg:
  """Base clearance above local terrain.

  ``reduction="mean"``, not ``"min"``: a single boulder under the belly should not read as
  the robot having collapsed. This feeds both the base-height reward and the low-clearance
  termination -- the terrain-relative replacement for ``root_height_below_minimum``, which
  reads absolute world z and is unusable on generated terrain.
  """
  return TerrainHeightSensorCfg(
    name=BASE_HEIGHT_SENSOR,
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=RingPatternCfg.single_ring(radius=0.12, num_samples=8),
    reduction="mean",
    max_distance=2.0,
    include_geom_groups=(0,),
  )


def _body_contact_sensor_cfg(name: str, geom_names: tuple[str, ...]) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=POSITION_TIMING.decimation,
  )


def get_body_contact_sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
  """Thigh / shank / trunk vs terrain.

  Thigh and shank contacts are *penalised* rather than terminating: on jagged ground a Go2
  brushes its legs constantly, and mjlab's Go1 rough task (which does terminate on thigh
  contact) is tuned for a tamer terrain mix. Only the trunk terminates.
  """
  return (
    _body_contact_sensor_cfg(THIGH_CONTACT_SENSOR, THIGH_COLLISION_GEOMS),
    _body_contact_sensor_cfg(SHANK_CONTACT_SENSOR, SHANK_COLLISION_GEOMS),
    _body_contact_sensor_cfg(TRUNK_CONTACT_SENSOR, TRUNK_COLLISION_GEOMS),
  )


def check_spec() -> mujoco.MjModel:
  """Assert the PD build is what we think it is.

  Unlike the flat task's check this cannot inspect ``get_spec()`` directly -- that spec has
  zero actuators by design, and the ``<position>`` elements only exist after mjlab's
  ``edit_spec`` runs. So build the Entity and assert on the compiled model.
  """
  from mjlab.entity import Entity

  entity = Entity(get_go2_robot_cfg())
  m = entity.spec.compile()

  assert (m.nu, m.nq, m.nv) == (12, 19, 18), f"topology changed: {m.nu=} {m.nq=} {m.nv=}"
  assert abs(m.body_subtreemass[1] - BASE_MASS) < 1e-3, (
    f"total mass {m.body_subtreemass[1]} != BASE_MASS {BASE_MASS}"
  )
  assert m.geom_margin.max() == 0.0, "contact margin must be 0"

  for jt, group in GO2_ACTUATORS.items():
    for leg in FEET:
      aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{jt}_joint")
      assert aid >= 0, f"no position actuator for {leg}_{jt}_joint"
      # mjBIAS_AFFINE proves this is a <position> element and not a leftover <motor>.
      assert m.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE, (
        f"{leg}_{jt}_joint is not a position actuator -- <motor> block not deleted?"
      )
      assert abs(m.actuator_gainprm[aid, 0] - group.kp) < 1e-9, f"{jt} kp"
      assert abs(m.actuator_biasprm[aid, 1] + group.kp) < 1e-9, f"{jt} -kp"
      assert abs(m.actuator_biasprm[aid, 2] + group.kd) < 1e-9, f"{jt} -kd"
      assert abs(m.actuator_forcerange[aid, 1] - group.effort_limit) < 1e-9, f"{jt} force"

  for name in FOOT_GEOMS:
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, f"no geom {name}"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}_site") >= 0
  assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer") >= 0

  for sensor_cfg in get_body_contact_sensor_cfgs():
    for geom_name in sensor_cfg.primary.pattern:
      assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0, (
        f"collision geom {geom_name} not found -- has go2_mjcf renamed its geoms?"
      )
  return m
