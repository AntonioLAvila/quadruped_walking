from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector
)

### A1 state
# position: [quat, translation, joints]
# velovity: [rotational, translational, joints]

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
