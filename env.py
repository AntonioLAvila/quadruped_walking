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
    InputPortIndex,
    OutputPortIndex,
    PortDataType,
    OutputPort,
    InputPort
)
import numpy as np
from pydrake.gym import DrakeGymEnv
from gymnasium.spaces import Box
from util import A1_q0, time_step, torque_scale
from typing import Callable, Optional, Type
from gymnasium import Env


class ObservationExtractor(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        # NOTE 12 joint positions, 3 body rotational velocities, 3 body translational, 12 joint velocities
        # in that order
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
    torque_multiplier = TorqueMultiplier(plant, scale=torque_scale)
    builder.AddNamedSystem('torque_multiplier', torque_multiplier)
    observation_extractor = obs_extractor(plant)
    builder.AddNamedSystem('observation_extractor', observation_extractor)

    builder.Connect(torque_multiplier.output_port, plant.get_actuation_input_port())
    builder.Connect(plant.get_state_output_port(), observation_extractor.input_port)

    builder.ExportInput(torque_multiplier.input_port) # 12
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

    _, plant, _ = make_a1_diagram(ObservationExtractor)

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
        reward=reward_fn,
        hardware=True
    )

    return env, meshcat


class A1_Env(Env):
    def __init__(
        self,
        observation_extractor: Type[ObservationExtractor],
        time_step: float,
        visualize: bool = False,
    ):
        super().__init__()
        self.meshcat = None
        if visualize:
            self.meshcat: Meshcat = StartMeshcat()
        self.observation_extractor = observation_extractor
        self.action_port: InputPort = InputPortIndex(0)
        self.observation_port: OutputPort = OutputPortIndex(0)
        self.simulator: Simulator = None
        self.generator: RandomGenerator = None
        self.time_step = time_step

    def simulation_factory(self, generator: RandomGenerator) -> Simulator:
        diagram, _, _ = make_a1_diagram(ObservationExtractor, meshcat=meshcat)
        simulator = Simulator(diagram)
        simulator.set_monitor(self.make_termination_conditions(diagram))
        simulator.Initialize()
        return simulator
    
    def _setup(self):
        """Completes the setup once we have a self.simulator."""
        system = self.simulator.get_system()

        # Setup action port
        if self.action_port.get_data_type() == PortDataType.kVectorValued:
            assert np.array_equal(self.action_space.shape, [self.action_port.size()])

        # Setup observation port
        if self.observation_port.get_data_type() == PortDataType.kVectorValued:
            assert np.array_equal(self.observation_space.shape, [self.observation_port.size()])
        
        def get_output_port(id):
            if isinstance(id, OutputPortIndex):
                return system.get_output_port(id)
            return system.GetOutputPort(id)

        # Setup reward
        if self.reward_port_id:
            reward_port = get_output_port(self.reward_port_id)
            self.reward = lambda system, context: reward_port.Eval(context)[0]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, any]:
        '''
        Takes an action and returns a tuple:
        (observation, reward, terminated, truncated, info)
        '''
        assert self.simulator, "You must call reset() first"

        context = self.simulator.get_context()
        time = context.get_time()

        self.action_port.FixValue(context, action)
        truncated = False
        # Observation prior to advancing the simulation.
        prev_observation = self.observation_port.Eval(context)
        try:
            status = self.simulator.AdvanceTo(time + self.time_step)
        except RuntimeError as e:
            # TODO(JoseBarreiros-TRI) We don't currently check for the
            # error coming from the solver failing to converge.
            warnings.warn("Calling Done after catching RuntimeError:")
            warnings.warn(e.args[0])
            # Truncated is used when the solver failed to converge.
            # Note: this is different than the official use of truncated
            # in Gymnasium:
            # "Whether the truncation condition outside the scope of the MDP
            # is satisfied. Typically, this is a timelimit, but
            # could also be used to indicate an agent physically going out
            # of bounds."
            # We handle the solver failure to converge by returning
            # zero reward and the previous observation since the action
            # was not successfully applied. This comes at a cost of
            # an extra evaluation of the observation port.
            truncated = True
            terminated = False
            reward = 0
            # Do not call info handler, as the simulator has faulted.
            info = dict()

            return prev_observation, reward, terminated, truncated, info

        observation = self.observation_port.Eval(context)
        reward = self.reward(self.simulator.get_system(), context)
        terminated = (
            not truncated
            and (status.reason()
                 == SimulatorStatus.ReturnReason.kReachedTerminationCondition))
        info = self.info_handler(self.simulator)

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, option: Optional[dict] = None):
        '''
        returns a tuple: (observations, info)
        '''
        super.reset(seed=seed)
        self.generator = RandomGenerator(seed)
        self.simulator = self.simulation_factory(self.generator)
        self._setup()

        context = self.simulator.get_mutable_context()
        context.SetTime(0)
        self.simulator.Initialize()
        observations = self.observation_port.Eval(context)

        return observations, dict()

    def render(self):
        assert self.simulator, 'call reset first'
        self.simulator.get_system().ForcedPublish(self.simulator.get_context())
        return
    
    def make_termination_conditions(self, diagram: Diagram):
        # TODO do this correctly
        def termination_conditions(context: Context):
            if context.get_time() > 10:
                return EventStatus.ReachedTermination(diagram, 'time limit')
            return EventStatus.Succeeded()
        return termination_conditions

    def reward_fn(sim_diagram: Diagram, sim_context: Context) -> float:
        # TODO make this good
        plant: MultibodyPlant = sim_diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyContextFromRoot(sim_context)
        trunk = plant.GetBodyByName('trunk')

        state_view = namedview('state', plant.GetStateNames())
        plant_state = plant.get_state_output_port().Eval(plant_context)
        state = state_view(plant_state)

        reward = 0

        # Linear velocity xy
        v_xy_des = np.array([1.5, 0])
        v_xy = np.array([state.a1_trunk_vx, state.a1_trunk_vy])
        diff_v_xy = v_xy_des - v_xy
        reward += 4 * np.exp(-np.dot(diff_v_xy, diff_v_xy)/0.25)

        # Linear velocity z
        reward += -1 * state.a1_trunk_vz**2

        # Angular velocity xy
        omega_xy = np.array([state.a1_trunk_wx, state.a1_trunk_wy])
        reward += -0.05 * np.dot(omega_xy, omega_xy)

        # Angular velocity z
        reward += 0.5 * np.exp(-1 * state.a1_trunk_wz**2 / 0.25)

        # Orientation
        R_WT = trunk.EvalPoseInWorld(plant_context).rotation()
        world_z = np.array([0,0,1])
        trunk_z = R_WT.col(2)
        diff_o = world_z - trunk_z
        reward += np.dot(diff_o, diff_o)

        # Torque
        actuation = sim_diagram.get_input_port(0).Eval(sim_context)
        reward += -1e-4 * np.dot(actuation, actuation)

        # Heel collision
        heel_frames = [
            plant.GetFrameByName('FL_calf_joint_parent'),
            plant.GetFrameByName('FR_calf_joint_parent'),
            plant.GetFrameByName('RL_calf_joint_parent'),
            plant.GetFrameByName('RR_calf_joint_parent')
        ]
        for heel_frame in heel_frames:
            p_WKnee = heel_frame.CalcPoseInWorld(plant_context).translation()
            if p_WKnee[2] < 0.1:
                reward += -0.5

        return reward



if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()

    diagram, plant, a1 = make_a1_diagram(ObservationExtractor, meshcat=meshcat)

    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    input_port = diagram.get_input_port(0)
    input_port.FixValue(diagram_context, np.zeros(plant.num_actuators()))

    sim = Simulator(diagram, diagram_context)
    meshcat.StartRecording()
    sim.AdvanceTo(3)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    while True: ...
