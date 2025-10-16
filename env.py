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
    MeshcatVisualizer
)
import numpy as np


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
        "ground_collision",
        CoulombFriction(1.0, 1.0),
    )
    
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    plant.Finalize()
    diagram = builder.Build()

    return diagram, plant, a1

if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()

    diagram, plant, a1_model = make_a1(meshcat)
    simulator = Simulator(diagram)

    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    initial_pose = RigidTransform(
        RollPitchYaw(0, 0, 0),   # no rotation
        [0, 0, 1.0]              # x, y, z position
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
