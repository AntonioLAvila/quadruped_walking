import mujoco
import numpy as np
from robot_descriptions import a1_mj_description
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from typing import Optional

'''
Compare history vs noise in signal.
    - Pass noise generator
    - Pass something that can
        - define observation_space
        - extract history
'''

class ObservationExtractor():
    '''Base observation extractor class'''
    obs_space: Optional[Box] = None

    def __init__(self):
        if self.obs_ub is None:
            raise ValueError(f'{self.__class__.__name__} must define self.obs_ub')

    def calc_obs(self, something):
        raise NotImplementedError('Subclasses must implement calc_obs()')


# class BasicExtractor(ObservationExtractor):
#     def __init__(self):
#         self.obs_space = Box(
#             low=np.
#         )
#         super().__init__()

#     def calc_obs(self, something):
#         x = self.get_input_port().Eval(context)
#         obs = x[7:]
#         output.SetFromVector(obs)

class Go2Env(MujocoEnv):
    def __init__(
        random_generator: np.random.Generator,
        observation_extractor: ObservationExtractor
    ):
        super().__init__(
            model_path=a1_mj_description.MJCF_PATH,
            frame_skip=1, # sim steps per episode step
            observation_space=observation_extractor.obs_space
        )


if __name__ == '__main__':
    go2 = mujoco.MjModel.from_xml_path(a1_mj_description.MJCF_PATH)
    data = mujoco.MjData(go2)
    
    print("Number of joints:", go2.njnt)
    print("Number of actuators:", go2.nu)
    print("Number of bodies:", go2.nbody)
    print("Number of velocities:", go2.nv)

    joint_upper_limits = []
    joint_lower_limits = []
    for i in range(go2.njnt):
        name = mujoco.mj_id2name(go2, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = go2.jnt_type[i]  # 0=free, 1=ball, 2=slide, 3=hinge
        range_ = go2.jnt_range[i]
        joint_lower_limits.append(float(range_[0]))
        joint_upper_limits.append(float(range_[1]))
        damping = go2.dof_damping[i] if i < len(go2.dof_damping) else None
        
        print(f"Joint {i}: {name}, type={joint_type}, range={range_}, damping={damping}")

    print(joint_lower_limits, '\n', joint_upper_limits)
