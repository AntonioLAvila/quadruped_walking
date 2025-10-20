from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector
)

class ObservationExtractor(LeafSystem):
    # TODO add noise argagrgagrgragargragragaargargarrgrggrag
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self.plant = plant
        obs_dim = 12 + 12 + 3 + 3 # NOTE 12 joint positions, 12 joint velocities, 3 body translational, 3 body rotational velocities
        self.input_port = self.DeclareVectorInputPort("state", plant.num_multibody_states())
        self.output_port = self.DeclareVectorOutputPort("observation", BasicVector(obs_dim), self.calc_obs)

    def calc_obs(self, context, output):
        x = self.get_input_port().Eval(context)
        obs = x[7:]
        output.SetFromVector(obs)
