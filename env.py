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
    namedview,
    EventStatus,
    SimulatorStatus,
    RigidTransform,
    RotationMatrix
)
import numpy as np
from gymnasium.spaces import Box
from util import A1_q0, time_step, torque_scale, time_limit, speed
from typing import Optional, Type
from gymnasium import Env
import warnings
from robot_descriptions import a1_description


class ObservationExtractor(LeafSystem):
    '''Base observation extractor class'''
    obs_dim: Optional[int] = None
    obs_lb: Optional[np.ndarray] = None
    obs_ub: Optional[np.ndarray] = None
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        if not hasattr(self, "obs_dim"):
            self.obs_dim = None
        if not hasattr(self, "obs_lb"):
            self.obs_lb = None
        if not hasattr(self, "obs_ub"):
            self.obs_ub = None

        if self.obs_dim is None:
            raise ValueError(f'{self.__class__.__name__} must define self.obs_dim before super().__init__()')
        if self.obs_lb is None:
            raise ValueError(f'{self.__class__.__name__} must define self.obs_lb')
        if self.obs_ub is None:
            raise ValueError(f'{self.__class__.__name__} must define self.obs_ub')
        
        self.input_port = self.DeclareVectorInputPort('state', plant.num_multibody_states())
        self.output_port = self.DeclareVectorOutputPort('observation', BasicVector(self.obs_dim), self.calc_obs)

    def calc_obs(self, context: Context, output: BasicVector):
        raise NotImplementedError('Subclasses must implement calc_obs()')


class BasicExtractor(ObservationExtractor):
    def __init__(self, plant: MultibodyPlant):
        # NOTE 12 joint positions, 3 body rotational velocities, 3 body translational, 12 joint velocities
        # in that order
        self.obs_dim = 12 + 12 + 3 + 3
        self.obs_lb = np.concatenate((
            plant.GetPositionLowerLimits()[7:],
            plant.GetVelocityLowerLimits()
        ))
        self.obs_ub = np.concatenate((
            plant.GetPositionUpperLimits()[7:],
            plant.GetVelocityUpperLimits()
        ))
        super().__init__(plant)

    def calc_obs(self, context: Context, output: BasicVector):
        x = self.get_input_port().Eval(context)
        obs = x[7:]
        output.SetFromVector(obs)


class TorqueMultiplier(LeafSystem):
    def __init__(self, plant: MultibodyPlant, scale=33.5):
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
    obs_extractor: Type[ObservationExtractor],
    static_friction=1.0, # rubber on concrete
    dynamic_friction=0.8,
    meshcat: Meshcat = None
) -> tuple[Diagram, MultibodyPlant, ModelInstanceIndex]:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    plant: MultibodyPlant = plant
    scene_graph: SceneGraph = scene_graph
    parser = Parser(plant, scene_graph)

    # add a1 (somewhat official from mujoco menagerie)
    # a1 = parser.AddModels('assets/a1.urdf')[0]
    parser.package_map().Add('a1_description', a1_description.PACKAGE_PATH)
    a1 = parser.AddModels(a1_description.URDF_PATH)[0]

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
    plant.SetDefaultPositions(A1_q0.copy())
    plant.SetDefaultFreeBodyPose(plant.GetBodyByName('base'), RigidTransform(RotationMatrix.Identity(), A1_q0[4:7]))

    # wire up observation/actuation
    torque_multiplier = TorqueMultiplier(plant, scale=torque_scale)
    builder.AddNamedSystem('torque_multiplier', torque_multiplier)
    observation_extractor = obs_extractor(plant)
    builder.AddNamedSystem('observation_extractor', observation_extractor)

    builder.Connect(torque_multiplier.output_port, plant.get_actuation_input_port())
    builder.Connect(plant.get_state_output_port(), observation_extractor.input_port)

    builder.ExportInput(torque_multiplier.input_port) # 12
    builder.ExportOutput(observation_extractor.output_port)

    # visualization options
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        # ContactVisualizer.AddToBuilder(builder, plant, meshcat)

    diagram = builder.Build()

    return diagram, plant, a1


class A1_Env(Env):
    def __init__(
        self,
        observation_extractor: Type[ObservationExtractor],
        visualize: bool = False,
    ):
        super().__init__()
        self.time_step = time_step
        self.generator = None  # Will be initialized on first reset

        self.meshcat = StartMeshcat() if visualize else None
        
        self.diagram, self.plant, _ = make_a1_diagram(observation_extractor, meshcat=self.meshcat)
        self.state_view = namedview('state', self.plant.GetStateNames())
        self.simulator = Simulator(self.diagram)
        self.context = None
        self.plant_context = None

        self.action_port = self.diagram.get_input_port(0)
        self.observation_port = self.diagram.get_output_port(0)

        # self.action_space = Box(
        #     low=np.full(self.plant.num_actuators(), -1.0),
        #     high=np.full(self.plant.num_actuators(), 1.0),
        #     dtype=np.float64
        # )
        self.action_space = Box(
            low=np.ones(12)*-33.5,
            high=np.ones(12)*33.5,
            dtype=np.float64
        )

        obs_ext: ObservationExtractor = self.diagram.GetSubsystemByName('observation_extractor')
        self.observation_space = Box(
            low=obs_ext.obs_lb,
            high=obs_ext.obs_ub,
            dtype=np.float64
        )
        
        self.simulator.set_monitor(self.termination_conditions)
        # self.simulator.set_target_realtime_rate(1.0) # NOTE this exists

    def reset(self, *, seed: Optional[int] = None, option: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.generator = np.random.default_rng(seed)
        elif self.generator is None:
            self.generator = np.random.default_rng()

        fresh_context = self.diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyContextFromRoot(fresh_context)
        
        fresh_context.SetTime(0.)

        initial_positions = A1_q0.copy()
        position_noise = self.generator.uniform(-0.05, 0.05, self.plant.num_positions())
        position_noise[:7] = 0  # No noise on the floating base
        self.plant.SetPositions(plant_context, initial_positions + position_noise)
        self.plant.SetVelocities(plant_context, np.zeros(self.plant.num_velocities()))

        self.action_port.FixValue(fresh_context, np.zeros(12))

        self.simulator.reset_context(fresh_context)

        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        self.simulator.Initialize()

        # Get the initial observation
        observations = self.observation_port.Eval(self.context)

        info = self.info_handler()

        return observations, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, any]:
        assert self.context, "You must call reset() first"

        time = self.context.get_time()

        last_context = self.context.Clone()
        self.action_port.FixValue(self.context, action)
        truncated = False
        prev_observation = self.observation_port.Eval(self.context)

        try:
            status = self.simulator.AdvanceTo(time + self.time_step)
        except RuntimeError as e:
            warnings.warn("Calling Done after catching RuntimeError:")
            warnings.warn(e.args[0])
            truncated = True
            terminated = False
            reward = 0
            info = dict()
            return prev_observation, reward, terminated, truncated, info

        observation = self.observation_port.Eval(self.context)
        reward = self.reward_fn(self.diagram, self.context, last_context)
        self.last_action = action
        terminated = (
            not truncated
            and (status.reason() == SimulatorStatus.ReturnReason.kReachedTerminationCondition)
        )
        info = self.info_handler()

        return observation, reward, terminated, truncated, info

    def info_handler(self) -> dict:
        return dict()

    def render(self):
        assert self.simulator, 'call reset first'
        self.diagram.ForcedPublish(self.context)
        return
    
    # TODO do this
    def termination_conditions(self, context: Context):
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(self.diagram, 'time limit')
        
        plant_context = self.plant.GetMyContextFromRoot(context)
        R_WT = self.plant.GetBodyByName('trunk').EvalPoseInWorld(plant_context).rotation()
        cos_angle = np.dot(np.array([0,0,1]), R_WT.col(2))
        if cos_angle < 0.25:
            return EventStatus.ReachedTermination(self.diagram, 'bad orientation')
        
        p_WT = self.plant.GetBodyByName('trunk').EvalPoseInWorld(plant_context).translation()
        if p_WT[2] < 0.1:
            return EventStatus.ReachedTermination(self.diagram, 'too low')

        return EventStatus.Succeeded()

    # TODO make this good
    def reward_fn(self, sim_diagram: Diagram, sim_context: Context, last_sim_context: Context) -> float:
        plant: MultibodyPlant = sim_diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyContextFromRoot(sim_context)
        last_plant_context = plant.GetMyMutableContextFromRoot(last_sim_context)
        trunk = plant.GetBodyByName('trunk')

        plant_state = plant.get_state_output_port().Eval(plant_context)
        state = self.state_view(plant_state)
        last_plant_state = plant.get_state_output_port().Eval(last_plant_context)
        last_state = self.state_view(last_plant_state)

        reward = 0

        # Linear velocity xy
        velocity_weight = 2.0
        v_xy_des = np.array([speed, 0])
        v_xy = np.array([state.a1_base_vx, state.a1_base_vy])
        diff_v_xy = v_xy_des - v_xy
        reward += velocity_weight * np.exp(-np.dot(diff_v_xy, diff_v_xy)/0.25)

        # Linear velocity z
        z_velocity_weight = -2.0
        reward += z_velocity_weight * state.a1_base_vz**2

        # Angular velocity xy
        xy_angular_velocity_weight = -0.05
        omega_xy = np.array([state.a1_base_wx, state.a1_base_wy])
        reward += xy_angular_velocity_weight * np.dot(omega_xy, omega_xy)

        # Angular velocity z
        z_angular_velocity_weight = 0.5
        reward += z_angular_velocity_weight * np.exp(-state.a1_base_wz**2/0.25)

        # Orientation
        orientation_weight = -1.0
        R_WT = trunk.EvalPoseInWorld(plant_context).rotation()
        trunk_z = R_WT.col(2)
        trunk_z_xy = trunk_z[:2]
        reward += orientation_weight * np.dot(trunk_z_xy, trunk_z_xy)

        # Torque
        torque_weight = -2e-4
        actuation = sim_diagram.get_input_port(0).Eval(sim_context)
        reward += torque_weight * np.dot(actuation, actuation)

        # Joint acceleration
        joint_acc_weight = -2.5e-8
        reward += joint_acc_weight * np.square(last_state[-7:] - state[-7:]).sum()

        # Action rate
        action_rate_weight = -0.01
        last_actuation = sim_diagram.get_input_port(0).Eval(last_sim_context)
        reward += action_rate_weight * np.square(last_actuation - actuation).sum()

        # # Foot air time
        # foot_frames = [
        #     plant.GetFrameByName('FL_foot'),
        #     plant.GetFrameByName('FR_foot'),
        #     plant.GetFrameByName('RL_foot'),
        #     plant.GetFrameByName('RR_foot')
        # ]
        # foot_min_clearance_height = 0.05
        # foot_clearance_weight = -5.0
        # for foot_frame in foot_frames:
        #     p_WFoot = foot_frame.CalcPoseInWorld(plant_context).translation()
        #     if p_WFoot[2] < foot_min_clearance_height:
        #         reward += foot_clearance_weight * (foot_min_clearance_height - p_WFoot[2])

        # Heel collision
        heel_clearance_height = 0.05
        heel_clearance_weight = -2
        heel_frames = [
            plant.GetFrameByName('FL_calf_joint_parent'),
            plant.GetFrameByName('FR_calf_joint_parent'),
            plant.GetFrameByName('RL_calf_joint_parent'),
            plant.GetFrameByName('RR_calf_joint_parent')
        ]
        for heel_frame in heel_frames:
            p_WKnee = heel_frame.CalcPoseInWorld(plant_context).translation()
            if p_WKnee[2] < heel_clearance_height:
                reward += heel_clearance_weight * (heel_clearance_height - p_WKnee[2])

        # # Stance feet penalty
        # contact_threshold = 0.01
        # stance_weight = -0.1
        # for foot_frame in foot_frames:
        #     p_WFoot = foot_frame.CalcPoseInWorld(plant_context).translation()
        #     if p_WFoot[2] < contact_threshold:
        #         V_WFoot = foot_frame.CalcSpatialVelocityInWorld(plant_context).translational()
        #         v_xy_foot = V_WFoot[:2]
        #         reward += stance_weight * np.dot(v_xy_foot, v_xy_foot)

        return reward

if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()

    diagram, plant, a1 = make_a1_diagram(BasicExtractor, meshcat=meshcat)

    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    # print(plant.GetStateNames())

    input_port = diagram.get_input_port(0)
    input_port.FixValue(diagram_context, np.zeros(plant.num_actuators()))

    sim = Simulator(diagram, diagram_context)
    meshcat.StartRecording()
    sim.AdvanceTo(3)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    while True: ...
