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
    R_from_quat,
    projected_gravity,
    weights
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
        # orientation, joint pos, body spatial vel, body rotational vel, joint vel, g_proj
        self.obs_space = Box(
            low=np.concatenate(([-1]*4, A1_joint_lb, [-np.inf]*18, [-1]*3)),
            high=np.concatenate(([1]*4, A1_joint_ub, [np.inf]*18, [1]*3)),
            dtype=np.float64
        )
        self.history_len = 0
        super().__init__()

    def calc_obs(self, mjdata, data_history: deque):
        o_and_q = mjdata.qpos[3:]
        g_proj = projected_gravity(mjdata.qpos[3:7])
        return np.concatenate((o_and_q, mjdata.qvel, g_proj))


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
        self.observation_extractor = observation_extractor
        self._max_episode_time = 30.0 # seconds
        self._upright = np.array([0,0,1])
        self._torque_scale = 1 # NOTE should maybe set to 10 idk we'll see
        self._v_xy_desired = np.array([1,0])
        self._desired_yaw_rate = 0.0
        self._contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]
        self._reset_rng = np.random.default_rng()

        # stateful things
        self._step = 0
        self._last_render_time = -1.0
        self._history_len = self.observation_extractor.history_len
        self._data_history = deque(maxlen=self._history_len) # oldest to youngest
        self._last_action = np.zeros(12) # NOTE this is seperate from _data_history because it's used only for reward calculation
        
    def step(self, action: np.ndarray):
        self._step += 1 # update for keeping time
        self._data_history.append(deepcopy(self.data)) # update history
        self.do_simulation(action*self._torque_scale, self.frame_skip)

        obs = self.observation_extractor.calc_obs(self.data, self._data_history)
        reward, reward_info = self._calc_reward(action)
        terminated, truncated = self._calc_term_trunc()
        info = {
            'p_WBody': self.data.qpos[:3],
            **reward_info
        }

        if self.render_mode == 'human' \
        and (self.data.time - self._last_render_time) > (1.0/self.metadata['render_fps']):
            self.render()
            self._last_render_time = self.data.time
        
        self._last_action = action.copy()

        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        # no noise on floating base
        noise = np.concatenate(([0]*7, self._reset_rng.uniform(-0.05, 0.05, 12)))
        self.data.qpos[:] = A1_q0 + noise

        self.data.ctrl[:] = self._reset_rng.normal(0, 0.1, 12)

        self._step = 0
        self._data_history.clear()
        self._last_render_time = -1.0
        self._last_action = np.zeros(12)

        observation = self.observation_extractor.calc_obs(self.data, self._data_history)
        return observation
        
    def _calc_reward(self, action: np.ndarray) -> tuple[float, dict]:
        # https://arxiv.org/abs/2203.05194 ignore the delta t
        reward = 0

        # Base velocity xy
        v_xy = self.data.qvel[:2]
        d_v_xy = self._v_xy_desired - v_xy
        reward += weights['base_v_xy'] * np.exp(-np.square(d_v_xy).sum()/weights['sigma_v_xy'])

        # Base velocity z
        v_z = self.data.qvel[2]
        reward += weights['base_v_z'] * v_z**2

        # Base angular velocity xy
        w_xy = self.data.qvel[3:5]
        reward += weights['angular_xy'] * np.square(w_xy).sum()

        # Base angular velocity z
        w_z = self.data.qvel[5]
        d_yaw = self._desired_yaw_rate - w_z
        reward += weights['yaw_rate'] * np.exp(-np.square(d_yaw).sum()/weights['sigma_yaw'])

        # Orientation
        quat = self.data.qpos[3:7]
        reward += weights['projected_gravity'] * np.square(projected_gravity(quat)[:2]).sum()

        # Effort
        reward += weights['effort'] * np.square(action).sum()

        # Joint accel
        qddot = self.data.qacc[-12:]
        reward += weights['joint_accel'] * np.square(qddot).sum()

        # Action rate
        d_action = self._last_action - action
        reward += weights['action_rate'] * np.square(d_action).sum()

        # Heel Contact
        is_contact = self.data.cfrc_ext[self._contact_indices] > 0.05
        reward += weights['heel_contact'] * np.sum(is_contact)

        # TODO add hip, thigh, air time rewards and fill out the dict probably lol

        # Alive
        reward += 1000

        return reward, dict()
    
    def _calc_term_trunc(self) -> tuple[bool, bool]:
        terminated = False
        
        body_quat = self.data.qpos[3:7]
        body_z_axis = R_from_quat(body_quat)[:, 2]
        cos_angle = np.dot(body_z_axis, self._upright)
        if cos_angle < 0.25:
            terminated = True # Bad orientation

        body_z = self.data.qpos[2]
        if body_z < 0.1:
            terminated = True # Fallen

        if not np.isfinite(self.state_vector()).all():
            terminated = True # Something bad happened

        truncated = self._step >= (self._max_episode_time / self.dt)

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

    print("Positions (qpos):", data.qpos)       # joint positions
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
    print(a1.opt.gravity)


    #=============MINIMAL RENDER===================
    env = A1_Env(BasicExtractor(), render_mode='human')
    obs, info = env.reset()

    env.mujoco_renderer.render("human")

    for _ in range(2000):
        obs, reward, terminated, truncated, info = env.step(np.random.default_rng().uniform(-33.5, 33.5, 12))
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

