import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import onnxruntime as ort
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    StartMeshcat,
    Meshcat,
    MeshcatVisualizer,
    Simulator,
    MultibodyPlant,
    SceneGraph,
    RigidTransform,
    ModelInstanceIndex,
    RigidBody,
    Box,
    ProximityProperties,
    CoulombFriction,
    AddRigidHydroelasticProperties,
    AddCompliantHydroelasticProperties,
    DiscreteContactApproximation,
    ContactModel,
    LeafSystem,
    Context,
    DiscreteValues,
    BasicVector,
    Saturation,
    Gain,
)
from robot_descriptions import go2_description

from go2_constants import (
    CTRL_DT,
    DEFAULT_HEIGHT,
    DEFAULT_JOINT_POS,
    GO2_ACTUATORS,
    JOINT_NAMES,
    JOINT_TORQUE_LIMITS_FLAT,
)

# Joint order used throughout training. The robot starts at the MJCF home height so
# the policy sees the initial condition it was reset to in training, instead of being
# asked to track a velocity while still in free fall.
JOINT_ORDER = JOINT_NAMES
DEFAULT_JOINT_POS = np.array(DEFAULT_JOINT_POS)
q0 = [1, 0, 0, 0] + [0, 0, DEFAULT_HEIGHT] + list(DEFAULT_JOINT_POS)
ACTION_SCALE = np.array(JOINT_TORQUE_LIMITS_FLAT)
# Every actuator parameter comes from GO2_ACTUATORS in go2_constants.py, the same
# table go2_robot.py feeds to mjlab -- so the two simulators cannot drift apart.
# Nothing actuator-related may be hard-coded in this file.
#
#   armature -> Drake reflected inertia (rotor_inertia * gear_ratio^2, gear_ratio 1).
#               MuJoCo adds it to the joint-space inertia diagonal; without it the
#               knee sees ~2.6x the angular acceleration for a given torque.
#   damping  -> Drake joint damping, matching MuJoCo's <joint damping>.
#   effort   -> explicit saturation: MuJoCo clamps ctrl to <motor ctrlrange>, but
#               Drake's effort_limit is advisory and unenforced on the actuation port.
#
# frictionloss has no MultibodyPlant equivalent and is deliberately not applied.
TORQUE_LIMITS = np.array(JOINT_TORQUE_LIMITS_FLAT)


def _joint_type(joint_name: str) -> str:
    """"FL_calf_joint" -> "calf"."""
    return joint_name.split("_")[1]
# Command held at zero for this long so the robot settles before it must track.
CMD_WARMUP_S = 0.5
CMD = np.array([1.25, 0.0, 0.0])
ONNX_POLICY_PATH = Path(__file__).parent / "logs/rsl_rl/go2_velocity/latest/model.onnx"


class ObservationExtractor(LeafSystem):
    def __init__(self, plant: MultibodyPlant, base_body_name: str = "base"):
        super().__init__()
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._base_body = plant.GetBodyByName(base_body_name)
        self._q_idx = [plant.GetJointByName(j).position_start() for j in JOINT_ORDER]
        self._v_idx = [plant.GetJointByName(j).velocity_start() for j in JOINT_ORDER]
        self._num_q = plant.num_positions()

        self.state_input = self.DeclareVectorInputPort('plant_state', plant.num_positions() + plant.num_velocities())
        # Wired to NNPolicy's action output, which is a pure function of that system's
        # discrete state. During NNPolicy's periodic update at time t, evaluating it
        # returns the pre-update state -- i.e. exactly a_{t-1}, which is what mjlab's
        # ``last_action`` term holds. Latching it into a *second* discrete state here
        # would delay it by another control step (obs would carry a_{t-2}).
        self.action_input = self.DeclareVectorInputPort('last_action', 12)

        # 42 = joint_pos 12 + joint_vel 12 + base_ang_vel 3 + gravity 3 + last_action 12.
        # (Was 45; base linear velocity is commented out of the actor obs in mjlab_env.py
        # because it is not reliably observable on the real Go2.)
        self.output_port = self.DeclareVectorOutputPort(
            'obs', 42, self._calc_obs,
            prerequisites_of_calc={self.state_input.ticket(), self.action_input.ticket()})

    def _calc_obs(self, context: Context, output: BasicVector):
        state = self.state_input.Eval(context)
        q, v = state[:self._num_q], state[self._num_q:]
        self._plant.SetPositionsAndVelocities(self._plant_context, state)

        joint_pos = q[self._q_idx] - DEFAULT_JOINT_POS
        joint_vel = v[self._v_idx]

        pose = self._plant.EvalBodyPoseInWorld(self._plant_context, self._base_body)
        R_inv = pose.rotation().inverse()
        spatial_vel = self._plant.EvalBodySpatialVelocityInWorld(self._plant_context, self._base_body)
        # base_lin_vel_b = R_inv @ spatial_vel.translational()  # not in the actor obs
        base_ang_vel_b = R_inv @ spatial_vel.rotational()
        g_proj_b = R_inv @ np.array([0.0, 0.0, -1.0])

        prev_action = self.action_input.Eval(context)

        output.SetFromVector(
            np.concatenate([
                joint_pos,
                joint_vel,
                # base_lin_vel_b,
                base_ang_vel_b,
                g_proj_b,
                prev_action,
            ]).astype(np.float32)
        )


class NNPolicy(LeafSystem):
    def __init__(self):
        super().__init__()
        self.session = ort.InferenceSession(str(ONNX_POLICY_PATH))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.obs_input = self.DeclareVectorInputPort('obs', 42)
        self.cmd_input = self.DeclareVectorInputPort('cmd', 3)

        self._action_state = self.DeclareDiscreteState(12)
        self.DeclarePeriodicDiscreteUpdateEvent(CTRL_DT, 0.0, self._update_action)

        self.output_port = self.DeclareVectorOutputPort('action', 12, self._calc_action, prerequisites_of_calc={self.xd_ticket()})

    def _update_action(self, context: Context, discrete_state: DiscreteValues):
        obs42 = self.obs_input.Eval(context)
        cmd = self.cmd_input.Eval(context)

        obs = np.concatenate([obs42, cmd]).astype(np.float32).reshape(1, -1)

        action = self.session.run([self.output_name], {self.input_name: obs})[0][0]
        discrete_state.set_value(self._action_state, action)

    def _calc_action(self, context: Context, output: BasicVector):
        output.SetFromVector(context.get_discrete_state(self._action_state).get_value())


class CommandSource(LeafSystem):
    """Velocity command [vx, vy, wz], held at zero for an initial settling window."""

    def __init__(self, command: np.ndarray, warmup_s: float):
        super().__init__()
        self._command = np.asarray(command, dtype=float)
        self._warmup_s = warmup_s
        self.output_port = self.DeclareVectorOutputPort('cmd', 3, self._calc_cmd)

    def _calc_cmd(self, context: Context, output: BasicVector):
        active = context.get_time() >= self._warmup_s
        output.SetFromVector(self._command if active else np.zeros(3))


def add_floor(
    plant: MultibodyPlant,
    dims: tuple[float, float, float],
    mu_static: float,
    mu_dynamic: float,
    color=[0.7, 0.5, 0.3, 1.0],
    contact_type='rigid'
) -> tuple[RigidBody, ModelInstanceIndex]:
    model = plant.AddModelInstance('floor_model')
    body = plant.AddRigidBody('floor_body', model)
    shape = Box(*dims)
    pose = RigidTransform([0, 0, -dims[-1]/2])

    contact_properties = ProximityProperties()
    contact_properties.AddProperty('material', 'coulomb_friction', CoulombFriction(mu_static, mu_dynamic))
    if contact_type == 'rigid':
        AddRigidHydroelasticProperties(0.05, contact_properties)
    elif contact_type == 'compliant':
        AddCompliantHydroelasticProperties(0.05, 1e7, contact_properties)
    else:
        raise RuntimeError(f'Contact type {contact_type} not supported')

    plant.RegisterVisualGeometry(body, pose, shape, 'floor_visual', color)
    plant.RegisterCollisionGeometry(
        body,
        pose,
        shape,
        "floor_collision",
        contact_properties
    )
    plant.WeldFrames(plant.world_frame(), body.body_frame())

    return body, model


def make_environment(meshcat: Meshcat) -> tuple[DiagramBuilder, MultibodyPlant, ModelInstanceIndex]:
    builder = DiagramBuilder()

    plant: MultibodyPlant
    scene_graph: SceneGraph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    parser = Parser(plant)
    parser.package_map().PopulateFromFolder(go2_description.PACKAGE_PATH)
    go2_model, = parser.AddModels(go2_description.URDF_PATH)

    # Add actuators and damping coefficients in drake, from the shared table.
    tree = ET.parse(go2_description.URDF_PATH)
    for joint_elem in tree.getroot().findall("joint"):
        if joint_elem.get("type") != "revolute":
            continue
        name = joint_elem.get("name")
        group = GO2_ACTUATORS[_joint_type(name)]
        actuator = plant.AddJointActuator(
            name, plant.GetJointByName(name), effort_limit=group.effort_limit)
        plant.GetJointByName(name).set_default_damping(group.damping)
        actuator.set_default_rotor_inertia(group.armature)

    # Wide enough that a 30 s run at ~1.3 m/s does not walk off the edge (the old
    # 50 m floor ran out at ~19 s, which looked exactly like a late-onset fall).
    floor_body, floor_model = add_floor(plant, (200, 200, 0.5), 0.7, 0.5)

    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.set_contact_model(ContactModel.kHydroelasticWithFallback)

    plant.Finalize()

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    return builder, plant, go2_model


if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()

    # make default environment
    builder, plant, go2_model = make_environment(meshcat)

    # Add the controller and its observation pipeline.
    controller = NNPolicy()
    builder.AddNamedSystem('nnpolicy', controller)

    extractor = ObservationExtractor(plant)
    builder.AddNamedSystem('obs_extractor', extractor)

    cmd_source = CommandSource(CMD, CMD_WARMUP_S)
    builder.AddNamedSystem('cmd_source', cmd_source)

    action_to_torque = builder.AddNamedSystem('action_to_torque', Gain(k=ACTION_SCALE))
    # Stand in for MuJoCo's <motor ctrlrange> clamp, which Drake does not apply.
    torque_limit = builder.AddNamedSystem(
        'torque_limit', Saturation(min_value=-TORQUE_LIMITS, max_value=TORQUE_LIMITS))

    # Connect things
    builder.Connect(plant.get_state_output_port(), extractor.state_input)
    builder.Connect(controller.output_port, extractor.action_input)
    builder.Connect(extractor.output_port, controller.obs_input)
    builder.Connect(cmd_source.output_port, controller.cmd_input)
    builder.Connect(controller.output_port, action_to_torque.get_input_port())
    builder.Connect(action_to_torque.get_output_port(), torque_limit.get_input_port())
    builder.Connect(torque_limit.get_output_port(), plant.get_actuation_input_port())

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    plant.SetPositions(plant_context, q0)

    sim = Simulator(diagram, diagram_context)
    sim.set_target_realtime_rate(1.0)
    meshcat.StartRecording()
    sim.AdvanceTo(10.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    input("Recording published. Press Enter to exit...")


    

