import mujoco
import numpy as np
from robot_descriptions import a1_mj_description
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from typing import Optional
from copy import deepcopy
from collections import deque
from util import (
    A1_joint_lb,
    A1_joint_ub,
    dt_control,
    dt_sim, A1_q0,
    default_cam_config,
    control_freq,
    euler_from_quaternion
)



class ObservationExtractor():
    '''Base observation extractor class'''
    obs_space: Optional[Box] = None
    history_len: int = None

    def __init__(self):
        if self.obs_space is None:
            raise ValueError(f'{self.__class__.__name__} must define self.obs_ub')
        if self.history_len is None:
            raise ValueError(f'{self.__class__.__name__} must define self.history_len')

    def calc_obs(self, mjdata, data_history: deque):
        # data_history is oldest to youngest
        raise NotImplementedError('Subclasses must implement calc_obs()')


class BasicExtractor(ObservationExtractor):
    def __init__(self):
        # o, q, v, w, qd
        self.obs_space = Box(
            low=np.concatenate(([-1]*4, A1_joint_lb, [-np.inf]*18)),
            high=np.concatenate(([1]*4, A1_joint_ub, [np.inf]*18)),
            dtype=np.float64
        )
        self.history_len = 0
        super().__init__()

    def calc_obs(self, mjdata, data_history: deque):
        o_and_q = mjdata.qpos[3:]
        return np.concatenate((o_and_q, mjdata.qvel))


class A1_Env(MujocoEnv):

    metadata = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array'
        ],
        'render_fps': control_freq
    }

    def __init__(
        self,
        observation_extractor: ObservationExtractor,
        **kwargs
    ):
        super().__init__(
            model_path=a1_mj_description.MJCF_PATH,
            frame_skip=int(dt_control/dt_sim),
            observation_space=observation_extractor.obs_space,
            default_camera_config=default_cam_config,
            **kwargs
        )
        self._max_episode_time = 30.0 # seconds
        self._step = 0
        self._last_render_time = -1.0

        self._reset_rng = np.random.default_rng()

        self.observation_extractor = observation_extractor
        self._history_len = self.observation_extractor.history_len
        self._data_history = deque(maxlen=self._history_len) # oldest to youngest
        
    def step(self, action):
        self._step += 1 # update for keeping time
        self._data_history.append(deepcopy(self.data)) # update history
        self.do_simulation(action, self.frame_skip)

        obs = self.observation_extractor.calc_obs(self.data, self._data_history)
        reward, reward_info = self._calc_reward(action)
        terminated, truncated = self._calc_term_trunc(action)
        info = {
            'p_WTrunk': self.data.qpos[:3],
            **reward_info
        }

        if self.render_mode == 'human' \
        and (self.data.time - self._last_render_time) > (1.0/self.metadata['render_fps']):
            self.render()
            self._last_render_time = self.data.time

        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        if self.render_mode == "human":
            self.mujoco_renderer.render("human")

        # no noise on floating base
        noise = np.concatenate(([0]*7, self._reset_rng.uniform(-0.05, 0.05, 12)))
        self.data.qpos[:] = A1_q0 + noise

        self.data.ctrl[:] = self._reset_rng.normal(0, 0.1, 12)

        self._step = 0
        self._data_history.clear()
        self._last_render_time = -1.0

        observation = self.observation_extractor.calc_obs(self.data, self._data_history)
        return observation
        
    def _calc_reward(self, action):
        # TODO calc reward
        return 1, dict()
    
    def _calc_term_trunc(self, action):
        # TODO do this
        terminated = self._step >= (self._max_episode_time / self.dt)
        truncated = False
        return terminated, truncated


if __name__ == '__main__':
    #=========INFO=========
    a1 = mujoco.MjModel.from_xml_path(a1_mj_description.MJCF_PATH)
    print(a1_mj_description.MJCF_PATH)
    data = mujoco.MjData(a1)
    
    print("Number of joints:", a1.njnt)
    print("Number of actuators:", a1.nu)
    print("Number of bodies:", a1.nbody)
    print("Number of velocities:", a1.nv)

    print("Positions (qpos):", data.qpos)        # joint positions
    print("Velocities (qvel):", data.qvel)      # joint velocities
    print("Accelerations (qacc):", data.qacc)   # joint accelerations
    print("Actuator forces (ctrl):", data.ctrl) # actuator commands

    print('model time step', a1.opt.timestep)

    joint_upper_limits = []
    joint_lower_limits = []
    for i in range(a1.njnt):
        name = mujoco.mj_id2name(a1, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = a1.jnt_type[i]  # 0=free, 1=ball, 2=slide, 3=hinge
        range_ = a1.jnt_range[i]
        joint_lower_limits.append(float(range_[0]))
        joint_upper_limits.append(float(range_[1]))
        damping = a1.dof_damping[i] if i < len(a1.dof_damping) else None
        
        print(f"Joint {i}: {name}, type={joint_type}, range={range_}, damping={damping}")

    print(joint_lower_limits, '\n', joint_upper_limits)
    print(a1.jnt_limited)


    #=============MINIMAL RENDER===================
    env = A1_Env(BasicExtractor(), render_mode='human')
    obs, info = env.reset()

    # Force render at start
    env.mujoco_renderer.render("human")

    for _ in range(2000):
        obs, reward, terminated, truncated, info = env.step(np.random.default_rng().uniform(-33.5, 33.5, 12))
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

