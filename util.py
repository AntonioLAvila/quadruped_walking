from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector
)

class ObservationExtractor(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self.plant = plant
        obs_dim = plant.num_positions() + plant.num_velocities()
        self.DeclareVectorInputPort("state", plant.num_multibody_states())
        self.DeclareVectorOutputPort("observation", BasicVector(obs_dim), self.calc_obs)

    def calc_obs(self, context, output):
        x = self.get_input_port().Eval(context)
        q = x[:self.plant.num_positions()]
        output.SetFromVector(q)
