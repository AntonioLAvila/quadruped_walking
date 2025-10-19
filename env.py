from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Simulator,
    RigidTransform,
    RollPitchYaw,
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
    RandomGenerator
)
import numpy as np
from pydrake.gym import DrakeGymEnv
from gymnasium.spaces import Box
from stable_baselines3 import PPO

def make_a1(meshcat: Meshcat) -> tuple[Diagram, MultibodyPlant, ModelInstanceIndex]:
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    plant: MultibodyPlant = plant
    scene_graph: SceneGraph = scene_graph

    parser = Parser(plant, scene_graph)
    a1 = parser.AddModels('assets/a1.urdf')[0]

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        HalfSpace.MakePose(np.array([0, 0, 1]), np.zeros(3)),
        HalfSpace(),
        'ground_collision',
        CoulombFriction(1.0, 1.0),
    )
    
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    plant.Finalize()

    builder.ExportInput(plant.get_actuation_input_port(), 'actuation') # 12
    builder.ExportOutput(plant.get_state_output_port(), 'state') # 37

    ContactVisualizer.AddToBuilder(builder, plant, meshcat)

    diagram = builder.Build()

    return diagram, plant, a1

def make_simulation_maker(meshcat: Meshcat):

    def simulation_factory(generator: RandomGenerator) -> Simulator:
        diagram, plant, a1 = make_a1(meshcat)
        simulator = Simulator(diagram)
        # TODO randomize stuff
        return simulator
    
    return simulation_factory

def reward(diagram: Diagram, context: Context) -> float:
    return 1


if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()

    diagram, plant, a1 = make_a1(meshcat)

    action_space = Box(
        low=plant.GetEffortLowerLimits(),
        high=plant.GetEffortUpperLimits(),
        dtype=np.float64
    )

    observation_space = Box(
        low=plant.GetPositionLowerLimits(),
        high=plant.GetPositionUpperLimits(),
        dtype=np.float64
    )

    env = DrakeGymEnv(
        simulator=make_simulation_maker(meshcat),
        time_step=0.1,
        action_space=action_space,
        observation_space=observation_space,
        reward=reward
    )

    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=10000)

    # diagram, plant, a1_model = make_a1()
    # simulator = Simulator(diagram)

    # context = simulator.get_mutable_context()
    # plant_context = plant.GetMyContextFromRoot(context)

    # initial_pose = RigidTransform(RollPitchYaw(0, 0, 0), [0, 0, 1.0])

    # plant.SetFreeBodyPose(
    #     plant_context,
    #     plant.GetBodyByName("trunk", a1_model),
    #     initial_pose
    # )

    # simulator.set_target_realtime_rate(1.0)
    # print("🌐 Open Meshcat at: http://localhost:7000")

    # meshcat.StartRecording()
    # simulator.AdvanceTo(5.0)
    # meshcat.PublishRecording()

    # while True:
    #     pass
