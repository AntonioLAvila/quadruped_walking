"""Sim-to-sim check of an exported rugged-task (PD position) ONNX policy in Drake.

    python scripts/verify_rugged.py [path/to/model.onnx]

This is a **flat-ground** check, deliberately. Putting mjlab's heightfield terrain into
Drake is a project, not a fix, and it is not what this script exists for. What it does
catch is everything between the policy and the robot: observation layout and ordering,
the history buffer, action scale and offset, the PD control law, joint ordering, and
whether the policy only walks because of MuJoCo-specific contact behaviour.

Run ``scripts/check_obs_layout.py`` first -- it proves the packer below matches mjlab
exactly. Without that, a wrong ordering here is silent garbage rather than an error.

Configuration is read from the ONNX metadata that mjlab embeds (joint names, action
scale, PD gains, default pose, observation term names), so this file cannot drift from
the trained policy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xml.etree.ElementTree as ET  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from pydrake.all import (  # noqa: E402
  AddMultibodyPlantSceneGraph,
  BasicVector,
  Box,
  CoulombFriction,
  Context,
  DiagramBuilder,
  DiscreteContactApproximation,
  DiscreteValues,
  LeafSystem,
  Meshcat,
  MeshcatVisualizer,
  ModelInstanceIndex,
  MultibodyPlant,
  Parser,
  ProximityProperties,
  RigidTransform,
  SceneGraph,
  Simulator,
  StartMeshcat,
)
from robot_descriptions import go2_description  # noqa: E402

from go2.constants import (  # noqa: E402
  DEFAULT_HEIGHT,
  DEFAULT_JOINT_POS,
  JOINT_ARMATURE_FLAT,
  JOINT_NAMES,
  JOINT_TORQUE_LIMITS_FLAT,
)
from go2.rugged.constants import (  # noqa: E402
  ACTOR_OBS_DIM,
  ACTOR_TERM_WIDTHS,
  FRAME_DIM,
  HISTORY_LENGTH,
  JOINT_KD_FLAT,
  JOINT_KP_FLAT,
  POSITION_ACTION_SCALE_FLAT,
  POSITION_TIMING,
)

DEFAULT_JOINT_POS = np.array(DEFAULT_JOINT_POS)
ACTION_SCALE = np.array(POSITION_ACTION_SCALE_FLAT)
KP = np.array(JOINT_KP_FLAT)
KD = np.array(JOINT_KD_FLAT)
TORQUE_LIMITS = np.array(JOINT_TORQUE_LIMITS_FLAT)
ARMATURE = np.array(JOINT_ARMATURE_FLAT)
CTRL_DT = POSITION_TIMING.ctrl_dt

OBS_DIM = ACTOR_OBS_DIM
TERM_WIDTHS = ACTOR_TERM_WIDTHS
CMD = np.array([1.0, 0.0, 0.0])
CMD_WARMUP_S = 0.5
PLANT_DT = 0.001

# The joint the URDF calls "damping" is unrelated to the PD kd -- it is the passive
# viscous term, matching POSITION_VISCOUS_DAMPING on the mjlab side.
PASSIVE_DAMPING = 0.5


def default_policy_path() -> Path:
  runs = sorted((Path(__file__).resolve().parents[1] / "logs/rsl_rl/go2_rugged").glob("*"))
  for run in reversed(runs):
    found = sorted(run.glob("*.onnx"))
    if found:
      return found[0]
  raise FileNotFoundError("no exported rugged ONNX policy found under logs/rsl_rl/go2_rugged")


def check_metadata(path: Path) -> None:
  """Cross-check the ONNX against this repo's constants, and fail loudly on drift."""
  model = onnx.load(str(path))
  meta = {p.key: p.value for p in model.metadata_props}

  dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
  assert dim == OBS_DIM, (
    f"policy expects {dim} observations but this script builds {OBS_DIM} "
    f"({FRAME_DIM} x H{HISTORY_LENGTH}) -- was HISTORY_LENGTH changed after training?"
  )

  names = [n for n, _ in TERM_WIDTHS]
  if "observation_names" in meta:
    trained = meta["observation_names"].split(",")
    assert trained == names, f"actor terms drifted: trained {trained}, packing {names}"

  def _floats(key: str) -> np.ndarray:
    return np.array([float(x) for x in meta[key].split(",")])

  if "joint_names" in meta:
    assert meta["joint_names"].split(",") == list(JOINT_NAMES), "joint order drifted"
  for key, ours in (
    ("action_scale", ACTION_SCALE),
    ("joint_stiffness", KP),
    ("joint_damping", KD),
  ):
    if key in meta:
      np.testing.assert_allclose(_floats(key), ours, rtol=1e-3, err_msg=f"{key} drifted")
  print(f"metadata OK: {dim}-dim obs, joint order and PD gains match GO2_ACTUATORS")


class PolicyObservation(LeafSystem):
  """Builds the 225-dim term-major observation the policy was trained on.

  Holds the H-1 *past* frames in discrete state and appends the live frame on the output
  port, so the vector the policy reads at time t genuinely ends at t. Pushing the current
  frame in the periodic update instead would leave the policy one frame stale -- the same
  off-by-one that used to affect ``last_action`` in the flat verification script.
  """

  def __init__(self, plant: MultibodyPlant, base_body_name: str = "base"):
    super().__init__()
    self._plant = plant
    self._plant_context = plant.CreateDefaultContext()
    self._base = plant.GetBodyByName(base_body_name)
    self._q_idx = [plant.GetJointByName(j).position_start() for j in JOINT_NAMES]
    self._v_idx = [plant.GetJointByName(j).velocity_start() for j in JOINT_NAMES]
    self._nq = plant.num_positions()

    self.state_input = self.DeclareVectorInputPort(
      "plant_state", plant.num_positions() + plant.num_velocities()
    )
    # Wired to the policy's own action output: during its update at time t that port
    # still holds a_{t-1}, which is exactly what mjlab's `last_action` term contains.
    self.action_input = self.DeclareVectorInputPort("last_action", 12)
    self.cmd_input = self.DeclareVectorInputPort("cmd", 3)

    self._past = self.DeclareDiscreteState((HISTORY_LENGTH - 1) * FRAME_DIM)
    self.DeclarePeriodicDiscreteUpdateEvent(CTRL_DT, 0.0, self._shift_history)

    self.output_port = self.DeclareVectorOutputPort(
      "obs",
      OBS_DIM,
      self._calc_obs,
      prerequisites_of_calc={
        self.state_input.ticket(),
        self.action_input.ticket(),
        self.cmd_input.ticket(),
        self.xd_ticket(),
      },
    )

  def frame(self, context: Context) -> np.ndarray:
    state = self.state_input.Eval(context)
    q, v = state[: self._nq], state[self._nq :]
    self._plant.SetPositionsAndVelocities(self._plant_context, state)

    pose = self._plant.EvalBodyPoseInWorld(self._plant_context, self._base)
    r_inv = pose.rotation().inverse()
    spatial = self._plant.EvalBodySpatialVelocityInWorld(self._plant_context, self._base)

    return np.concatenate(
      [
        q[self._q_idx] - DEFAULT_JOINT_POS,
        v[self._v_idx],
        r_inv @ spatial.rotational(),
        r_inv @ np.array([0.0, 0.0, -1.0]),
        self.action_input.Eval(context),
        self.cmd_input.Eval(context),
      ]
    ).astype(np.float32)

  def _shift_history(self, context: Context, discrete_state: DiscreteValues):
    past = context.get_discrete_state(self._past).get_value().reshape(-1, FRAME_DIM)
    shifted = np.roll(past, -1, axis=0)
    shifted[-1] = self.frame(context)
    discrete_state.set_value(self._past, shifted.reshape(-1))

  def _calc_obs(self, context: Context, output: BasicVector):
    past = context.get_discrete_state(self._past).get_value().reshape(-1, FRAME_DIM)
    frames = np.vstack([past, self.frame(context)])  # [H, 45], oldest -> newest
    # Term-major: all of term A's history, then all of term B's. See check_obs_layout.py.
    packed, offset = [], 0
    for _, width in TERM_WIDTHS:
      packed.append(frames[:, offset : offset + width].reshape(-1))
      offset += width
    output.SetFromVector(np.concatenate(packed).astype(np.float32))

  def backfill(self, root_context: Context) -> None:
    """Fill every history slot with the current frame, as mjlab does after a reset."""
    ctx = self.GetMyMutableContextFromRoot(root_context)
    tiled = np.tile(self.frame(ctx), HISTORY_LENGTH - 1)
    ctx.get_mutable_discrete_state(self._past).set_value(tiled)


class NNPolicy(LeafSystem):
  """ONNX actor, evaluated at the control rate."""

  def __init__(self, policy_path: Path):
    super().__init__()
    self.session = ort.InferenceSession(str(policy_path))
    self.input_name = self.session.get_inputs()[0].name
    self.output_name = self.session.get_outputs()[0].name

    self.obs_input = self.DeclareVectorInputPort("obs", OBS_DIM)
    self._action_state = self.DeclareDiscreteState(12)
    self.DeclarePeriodicDiscreteUpdateEvent(CTRL_DT, 0.0, self._update_action)
    self.output_port = self.DeclareVectorOutputPort(
      "action", 12, self._calc_action, prerequisites_of_calc={self.xd_ticket()}
    )

  def _update_action(self, context: Context, discrete_state: DiscreteValues):
    obs = np.asarray(self.obs_input.Eval(context), dtype=np.float32).reshape(1, -1)
    action = self.session.run([self.output_name], {self.input_name: obs})[0][0]
    discrete_state.set_value(self._action_state, action)

  def _calc_action(self, context: Context, output: BasicVector):
    output.SetFromVector(context.get_discrete_state(self._action_state).get_value())


class JointPD(LeafSystem):
  """The motor-board PD loop: tau = clip(kp*(target - q) - kd*qd, +/-effort).

  Evaluated at the **plant** rate, not the policy rate. That is the whole point of
  position control: the real Go2 closes this loop on the motor boards at ~1 kHz while the
  policy runs at 50 Hz, and collapsing the two would misrepresent the dynamics.
  """

  def __init__(self, plant: MultibodyPlant):
    super().__init__()
    self._plant = plant
    self._q_idx = [plant.GetJointByName(j).position_start() for j in JOINT_NAMES]
    self._v_idx = [plant.GetJointByName(j).velocity_start() for j in JOINT_NAMES]
    self._nq = plant.num_positions()

    self.state_input = self.DeclareVectorInputPort(
      "plant_state", plant.num_positions() + plant.num_velocities()
    )
    self.action_input = self.DeclareVectorInputPort("action", 12)
    self.output_port = self.DeclareVectorOutputPort("tau", 12, self._calc_tau)

  def _calc_tau(self, context: Context, output: BasicVector):
    state = self.state_input.Eval(context)
    q = state[: self._nq][self._q_idx]
    qd = state[self._nq :][self._v_idx]
    # JointPositionActionCfg(use_default_offset=True): target = action*scale + q_default.
    target = self.action_input.Eval(context) * ACTION_SCALE + DEFAULT_JOINT_POS
    tau = KP * (target - q) - KD * qd
    output.SetFromVector(np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS))


class CommandSource(LeafSystem):
  """Velocity command, held at zero for an initial settling window."""

  def __init__(self, command: np.ndarray, warmup_s: float):
    super().__init__()
    self._command = np.asarray(command, dtype=float)
    self._warmup_s = warmup_s
    self.output_port = self.DeclareVectorOutputPort("cmd", 3, self._calc)

  def _calc(self, context: Context, output: BasicVector):
    active = context.get_time() >= self._warmup_s
    output.SetFromVector(self._command if active else np.zeros(3))


def add_floor(plant: MultibodyPlant, dims=(200.0, 200.0, 0.5), mu=(0.7, 0.5)):
  model = plant.AddModelInstance("floor_model")
  body = plant.AddRigidBody("floor_body", model)
  shape = Box(*dims)
  pose = RigidTransform([0, 0, -dims[-1] / 2])
  props = ProximityProperties()
  props.AddProperty("material", "coulomb_friction", CoulombFriction(*mu))
  plant.RegisterVisualGeometry(body, pose, shape, "floor_visual", [0.7, 0.5, 0.3, 1.0])
  plant.RegisterCollisionGeometry(body, pose, shape, "floor_collision", props)
  plant.WeldFrames(plant.world_frame(), body.body_frame())
  return body, model


def make_environment(meshcat: Meshcat | None):
  builder = DiagramBuilder()
  plant: MultibodyPlant
  scene_graph: SceneGraph
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=PLANT_DT)

  parser = Parser(plant)
  parser.package_map().PopulateFromFolder(go2_description.PACKAGE_PATH)
  (go2_model,) = parser.AddModels(go2_description.URDF_PATH)

  tree = ET.parse(go2_description.URDF_PATH)
  for joint_elem in tree.getroot().findall("joint"):
    if joint_elem.get("type") != "revolute":
      continue
    name = joint_elem.get("name")
    index = list(JOINT_NAMES).index(name)
    actuator = plant.AddJointActuator(
      name, plant.GetJointByName(name), effort_limit=float(TORQUE_LIMITS[index])
    )
    # Reflected rotor inertia == MuJoCo's <joint armature> (gear ratio is 1). Without it
    # the knee sees ~2.6x the angular acceleration for a given torque.
    actuator.set_default_rotor_inertia(float(ARMATURE[index]))
    plant.GetJointByName(name).set_default_damping(PASSIVE_DAMPING)

  add_floor(plant)
  plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
  plant.Finalize()

  if meshcat is not None:
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
  return builder, plant, go2_model


def main() -> None:
  policy_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_policy_path()
  print(f"policy: {policy_path}")
  check_metadata(policy_path)

  meshcat = StartMeshcat()
  builder, plant, _ = make_environment(meshcat)

  policy = builder.AddNamedSystem("policy", NNPolicy(policy_path))
  observer = builder.AddNamedSystem("observer", PolicyObservation(plant))
  cmd = builder.AddNamedSystem("cmd", CommandSource(CMD, CMD_WARMUP_S))
  pd = builder.AddNamedSystem("pd", JointPD(plant))

  builder.Connect(plant.get_state_output_port(), observer.state_input)
  builder.Connect(policy.output_port, observer.action_input)
  builder.Connect(cmd.output_port, observer.cmd_input)
  builder.Connect(observer.output_port, policy.obs_input)
  builder.Connect(plant.get_state_output_port(), pd.state_input)
  builder.Connect(policy.output_port, pd.action_input)
  builder.Connect(pd.output_port, plant.get_actuation_input_port())

  diagram = builder.Build()
  context = diagram.CreateDefaultContext()
  plant_context = plant.GetMyMutableContextFromRoot(context)
  plant.SetPositions(
    plant_context, [1, 0, 0, 0] + [0, 0, DEFAULT_HEIGHT] + list(DEFAULT_JOINT_POS)
  )
  observer.backfill(context)

  sim = Simulator(diagram, context)
  sim.set_target_realtime_rate(1.0)
  meshcat.StartRecording()
  sim.AdvanceTo(10.0)
  meshcat.StopRecording()
  meshcat.PublishRecording()
  input("Recording published. Press Enter to exit...")


if __name__ == "__main__":
  main()
