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
    ConstantVectorSource,
    Gain,
)
from robot_descriptions import go2_description

q0 = [1, 0, 0, 0] + [0, 0, 0.3] + [0, 0.9, -1.8]*4
# Joint order used throughout training
JOINT_ORDER = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
DEFAULT_JOINT_POS = np.array([0.0, 0.9, -1.8] * 4)
ACTION_SCALE = np.array([23.7, 23.7, 45.43] * 4)
JOINT_DAMPING = 2.0
CTRL_DT = 0.005
ONNX_POLICY_PATH = Path(__file__).parent / "logs/rsl_rl/go2_velocity/best/2026-06-01_17-10-55.onnx"


class ObservationExtractor(LeafSystem):
    def __init__(self, plant: MultibodyPlant, base_body_name: str = "base"):
        super().__init__()
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._base_body = plant.GetBodyByName(base_body_name)
        self._q_idx = [plant.GetJointByName(j).position_start() for j in JOINT_ORDER]
        self._v_idx = [plant.GetJointByName(j).velocity_start() for j in JOINT_ORDER]
        self._num_q = plant.num_positions()

        self.state_input = self.DeclareVectorInputPort(
            'plant_state', plant.num_positions() + plant.num_velocities())
        self.action_input = self.DeclareVectorInputPort('last_action', 12)

        self._prev_action_state = self.DeclareDiscreteState(12)
        self.DeclarePeriodicDiscreteUpdateEvent(CTRL_DT, 0.0, self._latch_action)

        self.output_port = self.DeclareVectorOutputPort(
            'obs', 45, self._calc_obs,
            prerequisites_of_calc={self.state_input.ticket(), self.xd_ticket()})

    def _latch_action(self, context: Context, discrete_state: DiscreteValues):
        discrete_state.set_value(self._prev_action_state, self.action_input.Eval(context))

    def _calc_obs(self, context: Context, output: BasicVector):
        state = self.state_input.Eval(context)
        q, v = state[:self._num_q], state[self._num_q:]
        self._plant.SetPositionsAndVelocities(self._plant_context, state)

        joint_pos = q[self._q_idx] - DEFAULT_JOINT_POS
        joint_vel = v[self._v_idx]

        pose = self._plant.EvalBodyPoseInWorld(self._plant_context, self._base_body)
        R_inv = pose.rotation().inverse()
        spatial_vel = self._plant.EvalBodySpatialVelocityInWorld(self._plant_context, self._base_body)
        base_lin_vel_b = R_inv @ spatial_vel.translational()
        base_ang_vel_b = R_inv @ spatial_vel.rotational()
        g_proj_b = R_inv @ np.array([0.0, 0.0, -1.0])

        prev_action = context.get_discrete_state(self._prev_action_state).get_value()

        output.SetFromVector(np.concatenate([
            joint_pos, joint_vel, base_lin_vel_b, base_ang_vel_b, g_proj_b, prev_action,
        ]).astype(np.float32))


class NNPolicy(LeafSystem):
    def __init__(self):
        super().__init__()
        self.session = ort.InferenceSession(str(ONNX_POLICY_PATH))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.obs_input = self.DeclareVectorInputPort('obs', 45)
        self.cmd_input = self.DeclareVectorInputPort('cmd', 3)

        self._action_state = self.DeclareDiscreteState(12)
        self.DeclarePeriodicDiscreteUpdateEvent(CTRL_DT, 0.0, self._update_action)

        self.output_port = self.DeclareVectorOutputPort(
            'action', 12, self._calc_action,
            prerequisites_of_calc={self.xd_ticket()})

    def _update_action(self, context: Context, discrete_state: DiscreteValues):
        obs45 = self.obs_input.Eval(context)
        cmd = self.cmd_input.Eval(context)

        obs = np.concatenate([obs45, cmd]).astype(np.float32).reshape(1, -1)

        action = self.session.run([self.output_name], {self.input_name: obs})[0][0]
        discrete_state.set_value(self._action_state, action)

    def _calc_action(self, context: Context, output: BasicVector):
        output.SetFromVector(context.get_discrete_state(self._action_state).get_value())


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

    # Add actuators and dapming coefficients in drake
    tree = ET.parse(go2_description.URDF_PATH)
    for joint_elem in tree.getroot().findall("joint"):
        if joint_elem.get("type") != "revolute":
            continue
        name = joint_elem.get("name")
        effort = float(joint_elem.find("limit").get("effort"))
        plant.AddJointActuator(name, plant.GetJointByName(name), effort_limit=effort)
        plant.GetJointByName(name).set_default_damping(JOINT_DAMPING)

    floor_body, floor_model = add_floor(plant, (50, 50, 0.5), 0.9, 0.9)

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

    cmd_source = ConstantVectorSource(np.array([0.5, 0.0, 0.0]))
    builder.AddNamedSystem('cmd_source', cmd_source)

    action_to_torque = builder.AddNamedSystem('action_to_torque', Gain(k=ACTION_SCALE))

    # Connect things
    builder.Connect(plant.get_state_output_port(), extractor.state_input)
    builder.Connect(controller.output_port, extractor.action_input)
    builder.Connect(extractor.output_port, controller.obs_input)
    builder.Connect(cmd_source.get_output_port(), controller.cmd_input)
    builder.Connect(controller.output_port, action_to_torque.get_input_port())
    builder.Connect(action_to_torque.get_output_port(), plant.get_actuation_input_port())

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


    

