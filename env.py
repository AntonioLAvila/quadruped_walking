from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Simulator,
    FindResourceOrThrow,
    AddDefaultVisualization,
    RigidTransform,
    RollPitchYaw,
    CoulombFriction,
    HalfSpace,
    RobotDiagramBuilder,
    Diagram,
    MultibodyPlant,
    ModelInstanceIndex,
    StartMeshcat
)
import numpy as np


def make_a1(meshcat) -> tuple[Diagram, MultibodyPlant, ModelInstanceIndex]:
    '''
    returns the a1 diagram and a hook for the a1
    '''
    robot_builder = RobotDiagramBuilder(time_step=1e-4)
    plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()
    parser = robot_builder.parser()
    builder = robot_builder.builder()
    
    parser = Parser(plant)
    a1_model = parser.AddModels('assets/a1.urdf')[0]

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        HalfSpace.MakePose(np.array([0, 0, 1]), np.zeros(3)),
        HalfSpace(),
        "ground_collision",
        CoulombFriction(1.0, 1.0),
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        HalfSpace.MakePose(np.array([0, 0, 1]), np.zeros(3)),
        HalfSpace(),
        "ground_visual",
        [0.5, 0.5, 0.5, 1.0],
    )

    plant.Finalize()

    AddDefaultVisualization(builder, meshcat=meshcat)

    diagram = robot_builder.Build()

    return diagram, plant, a1_model


meshcat = StartMeshcat()

diagram, plant, a1_model = make_a1(meshcat)
simulator = Simulator(diagram)

context = simulator.get_mutable_context()
plant_context = plant.GetMyContextFromRoot(context)

initial_pose = RigidTransform(
    RollPitchYaw(0, 0, 0),   # no rotation
    [0, 0, 0.3]              # x, y, z position
)
plant.SetFreeBodyPose(
    plant_context,
    plant.GetBodyByName("trunk", a1_model),
    initial_pose
)

simulator.set_target_realtime_rate(1.0)
print("🌐 Open Meshcat at: http://localhost:7000")

meshcat.StartRecording()
simulator.AdvanceTo(5.0)
meshcat.PublishRecording()

while True:
    pass
