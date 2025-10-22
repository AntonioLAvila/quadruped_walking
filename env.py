from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Simulator,
    CoulombFriction,
    HalfSpace,
    Diagram,
    MultibodyPlant,
    ModelInstanceIndex,
    StartMeshcat,
    SceneGraph,
    Meshcat,
    MeshcatVisualizer,
    ContactVisualizer,
    Context,
    RandomGenerator,
    LeafSystem,
    BasicVector,
    AddFrameTriadIllustration
)
import numpy as np
from pydrake.gym import DrakeGymEnv
from gymnasium.spaces import Box
from util import A1_q0, standing_torque, time_step, torque_scaler
from typing import Callable, Optional


class ObservationExtractor(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        # NOTE 12 joint positions, 3 body rotational velocities, 3 body translational, 12 joint velocities
        # in that order
        # TODO maybe add gravity in the body frame
        obs_dim = 12 + 12 + 3 + 3 
        self.input_port = self.DeclareVectorInputPort('state', plant.num_multibody_states())
        self.output_port = self.DeclareVectorOutputPort('observation', BasicVector(obs_dim), self.calc_obs)

    def calc_obs(self, context: Context, output: BasicVector):
        # TODO add noise argagrgagrgragargragragaargargarrgrggrag
        x = self.get_input_port().Eval(context)
        obs = x[7:]
        output.SetFromVector(obs)


class TorqueMultiplier(LeafSystem):
    def __init__(self, plant: MultibodyPlant, scale=10):
        super().__init__()
        self.scale = scale
        self.lower_limits = plant.GetEffortLowerLimits()
        self.upper_limits = plant.GetEffortUpperLimits()
        self.input_port = self.DeclareVectorInputPort("NN_out", plant.num_actuators())
        self.output_port = self.DeclareVectorOutputPort("applied_torque", BasicVector(plant.num_actuators()), self.calc_torque)

    def calc_torque(self, context: Context, output: BasicVector):
        torque = self.input_port.Eval(context)
        scaled = self.scale*torque
        clamped = np.minimum(np.maximum(scaled, self.lower_limits), self.upper_limits)
        output.SetFromVector(clamped)


def make_a1_diagram(
    static_friction=1.0,
    dynamic_friction=1.0,
    meshcat: Meshcat = None
) -> tuple[Diagram, MultibodyPlant, ModelInstanceIndex]:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    plant: MultibodyPlant = plant
    scene_graph: SceneGraph = scene_graph
    parser = Parser(plant, scene_graph)

    # add a1
    a1 = parser.AddModels('assets/a1.urdf')[0]

    # register floor
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        HalfSpace.MakePose(np.array([0, 0, 1]), np.zeros(3)),
        HalfSpace(),
        'ground_collision',
        CoulombFriction(static_friction, dynamic_friction),
    )

    plant.Finalize()

    # set default position
    plant.SetDefaultPositions(A1_q0)

    # wire up observation/actuation
    torque_multiplier = TorqueMultiplier(plant, scale=torque_scaler)
    builder.AddNamedSystem('torque_multiplier', torque_multiplier)
    observation_extractor = ObservationExtractor(plant)
    builder.AddNamedSystem('observation_extractor', observation_extractor)

    builder.Connect(torque_multiplier.output_port, plant.get_actuation_input_port())
    builder.Connect(plant.get_state_output_port(), observation_extractor.input_port)

    builder.ExportInput(torque_multiplier.input_port) # 12
    builder.ExportOutput(observation_extractor.output_port) # 30


    # frames for debug
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     frame=plant.GetFrameByName('FL_calf_joint_parent')
    # )


    # visualization options
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        ContactVisualizer.AddToBuilder(builder, plant, meshcat)

    diagram = builder.Build()

    return diagram, plant, a1


def make_gym_env(
    reward_fn: Callable[[Diagram, Context], float],
    make_sim_maker: Callable[[Optional[Meshcat]], Callable[[RandomGenerator], Simulator]],
    visualize=False,
) -> tuple[DrakeGymEnv, Optional[Meshcat]]:
    meshcat = None
    if visualize:
        meshcat = StartMeshcat()

    _, plant, _ = make_a1_diagram()

    action_space = Box(
        low=plant.GetEffortLowerLimits(),
        high=plant.GetEffortUpperLimits(),
        dtype=np.float64
    )

    observation_space = Box(
        low=np.concatenate((
            plant.GetPositionLowerLimits()[7:],
            plant.GetVelocityLowerLimits()
        )),
        high=np.concatenate((
            plant.GetPositionUpperLimits()[7:],
            plant.GetVelocityUpperLimits()
        )),
        dtype=np.float64
    )

    env = DrakeGymEnv(
        simulator=make_sim_maker(meshcat),
        time_step=time_step,
        action_space=action_space,
        observation_space=observation_space,
        reward=reward_fn
    )

    return env, meshcat


def make_simulation_maker(meshcat: Meshcat = None):

    def simulation_factory(generator: RandomGenerator) -> Simulator:
        diagram, _, _ = make_a1_diagram(meshcat=meshcat)
        simulator = Simulator(diagram)
        # TODO randomize stuff agraagaragragragragragragrgragragra
        return simulator
    
    return simulation_factory


def reward_fn(diagram: Diagram, context: Context) -> float:
    # TODO make this good
    plant: MultibodyPlant = diagram.GetSubsystemByName('plant')
    plant_context = plant.GetMyContextFromRoot(context)
    trunk = plant.GetBodyByName('trunk')

    X_WT = trunk.EvalPoseInWorld(plant_context)
    R_WT = X_WT.rotation()
    p_WT = X_WT.translation()
    V_WT = trunk.EvalSpatialVelocityInWorld(plant_context)
    # V_WT = trunk.body_frame().CalcSpatialVelocity(plant_context, trunk.body_frame(), trunk.body_frame())
    v_WT = V_WT.translational()
    omega_WT = V_WT.rotational()
    actuation = diagram.get_input_port(0).Eval(context)
    joint_positions = plant.GetPositions(plant_context)[7:]

    reward = 0

    # Linear velocity
    v_xy_des = np.array([1.5, 0])
    v_xy = v_WT[:2]
    diff_v_xy = v_xy_des - v_xy
    reward += 2 * time_step * np.exp(-np.dot(diff_v_xy, diff_v_xy)/0.25)

    p_z = p_WT[2]
    diff_p_z = (0.3 - p_z)
    reward += -time_step * np.dot(diff_p_z, diff_p_z)
    # reward += -2 * time_step * v_z**2

    # Angular velocity
    omega_xy = omega_WT[:2]
    reward += -0.05 * time_step * np.dot(omega_xy, omega_xy)

    omega_z = omega_WT[2]
    reward += 0.5 * time_step * np.exp(-1 * omega_z**2 / 0.25)

    # Orientation
    world_z = np.array([0,0,1])
    trunk_z = R_WT.col(2)
    diff_o = world_z - trunk_z
    reward += -time_step * np.dot(diff_o, diff_o)

    # Torque
    # reward += -2e-4 * np.dot(actuation, actuation)

    # Joint positions
    for q, q_des in zip(joint_positions, A1_q0[7:]):
        reward += -time_step * (q_des - q)

    # Heel collision
    knee_frames = [
        plant.GetFrameByName('FL_calf_joint_parent'),
        plant.GetFrameByName('FR_calf_joint_parent'),
        plant.GetFrameByName('RL_calf_joint_parent'),
        plant.GetFrameByName('RR_calf_joint_parent')
    ]
    for knee_frame in knee_frames:
        p_WKnee = knee_frame.CalcPoseInWorld(plant_context).translation()
        if p_WKnee[2] < 0.1:
            reward += -time_step

    return reward


if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()

    diagram, plant, a1 = make_a1_diagram(meshcat=meshcat)

    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    trunk = plant.GetBodyByName('trunk')

    V_WT = trunk.body_frame().CalcSpatialVelocity(plant_context, trunk.body_frame(), trunk.body_frame()).translational()

    input_port = diagram.get_input_port(0)
    input_port.FixValue(diagram_context, np.zeros(plant.num_actuators()))

    sim = Simulator(diagram, diagram_context)
    meshcat.StartRecording()
    sim.AdvanceTo(3)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    while True: ...
