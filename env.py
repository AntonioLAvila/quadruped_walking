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
    BasicVector
)
import numpy as np
from pydrake.gym import DrakeGymEnv
from gymnasium.spaces import Box
from util import A1_q0, standing_torque, time_step
from typing import Callable, Optional


class ObservationExtractor(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self.plant = plant
        # NOTE 12 joint positions, 3 body rotational velocities, 3 body translational, 12 joint velocities
        # in that order
        # TODO maybe add gravity in the body frame
        obs_dim = 12 + 12 + 3 + 3 
        self.input_port = self.DeclareVectorInputPort("state", plant.num_multibody_states())
        self.output_port = self.DeclareVectorOutputPort("observation", BasicVector(obs_dim), self.calc_obs)

    def calc_obs(self, context, output):
        # TODO add noise argagrgagrgragargragragaargargarrgrggrag
        x = self.get_input_port().Eval(context)
        obs = x[7:]
        output.SetFromVector(obs)


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
    builder.ExportInput(plant.get_actuation_input_port(), 'actuation') # 12
    observation_extractor = ObservationExtractor(plant)
    builder.AddNamedSystem('observation_extractor', observation_extractor)
    builder.Connect(plant.get_state_output_port(), observation_extractor.input_port)
    builder.ExportOutput(observation_extractor.output_port) # 30

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
    reward = 0

    # move forward
    v_d = 1.0 # m/s
    v_WT = trunk.EvalSpatialVelocityInWorld(plant_context).translational()
    v_fwd = v_WT[0]
    reward += np.exp(-2.5 * (v_fwd - v_d)**2)

    X_WT = trunk.EvalPoseInWorld(plant_context)
    R_WT = X_WT.rotation()
    p_WT = X_WT.translation()

    # Orientation
    world_z = np.array([0,0,1])
    trunk_z = R_WT.col(2)
    reward += 2*np.dot(world_z, trunk_z)

    # Height
    z_d = 0.3
    reward += 0.5 * np.exp(-100 * (p_WT[2] - z_d)**2)

    # Effort
    # actuation = diagram.get_input_port(0).Eval(context)
    # actuation_error = np.sum(actuation - standing_torque)
    # reward += np.exp(-1 * (actuation_error)**2)
    actuation = diagram.get_input_port(0).Eval(context)
    reward += -1e-4 * np.sum(actuation)

    return reward


if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()

    diagram, plant, a1 = make_a1_diagram(meshcat)
    diagram_context = diagram.CreateDefaultContext()
    sim = Simulator(diagram, diagram_context)
    meshcat.StartRecording()
    sim.AdvanceTo(3)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    while True: ...
