from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector
)
import numpy as np

### A1 state
# position: [quat, translation, joints]
# velovity: [rotational, translational, joints]
A1_q0 = [1, 0, 0, 0] + [0, 0, 0.3] + [0, np.pi/4, -np.pi/2] * 4

standing_torque = [-0.02848178, 0.43561059, 4.29359326, -0.02848178, 0.38919193, 4.41591229, -0.04882372, 0.23277197, 4.6992705, -0.04882372, 0.19840704, 4.79748208]

class ObservationExtractor(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self.plant = plant
        # NOTE 12 joint positions, 3 body rotational velocities, 3 body translational, 12 joint velocities
        # in that order
        obs_dim = 12 + 12 + 3 + 3 
        self.input_port = self.DeclareVectorInputPort("state", plant.num_multibody_states())
        self.output_port = self.DeclareVectorOutputPort("observation", BasicVector(obs_dim), self.calc_obs)

    def calc_obs(self, context, output):
        # TODO add noise argagrgagrgragargragragaargargarrgrggrag
        x = self.get_input_port().Eval(context)
        obs = x[7:]
        output.SetFromVector(obs)
