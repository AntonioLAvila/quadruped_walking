import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from collections import deque
import os

'''
Credit to nimazareian on github for adapting mujoco menagerie Go1
MJCF scene and model for torque control.

jump run thing commit: 44b224419b5845415acfbd45b6917194eb22386f
'''

default_cam_config = {
    "azimuth": 90.0,
    "distance": 5.0,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.0, 0.0]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}


class Go1_Env(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array", 
            "depth_array"
        ]
    }

    def __init__(self, history_length=1, torque_scale=1, **kwargs):

        super().__init__(
            model_path=os.path.join(os.path.dirname(__file__), "unitree_go1", "scene_torque.xml"),
            frame_skip=5,  # 100 Hz
            observation_space=None,
            default_camera_config=default_cam_config,
            **kwargs,
        )

        self.metadata = {
            'render_modes': ['human', 'rgb_array', 'depth_array'],
            'render_fps': 60,
        }
        self._torque_scale = torque_scale
        self._history_len = history_length


        self._reset_rng = np.random.default_rng()
        self._max_episode_time = 15.0  # seconds
        self._gravity = np.array(self.model.opt.gravity)

        # Used in reward
        self._upright = np.array([0, 0, 1.0])
        self._v_xy_desired = np.array([1.5, 0])
        self._desired_yaw_rate = 0.0
        self._contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]  # hip and thigh
        self._hip_indices = [7, 10, 13, 16]
        self._thigh_indices = [8, 11, 14, 17]
        self._hip_q0 = self.model.key_qpos[0][self._hip_indices]
        self._thigh_q0 = self.model.key_qpos[0][self._thigh_indices]
        self._feet_indices = [4, 7, 10, 13]

        # Stateful things
        self._step = 0
        self._last_render_time = -1.0
        self._last_contacts = np.zeros(4)
        self._feet_air_time = np.zeros(4)
        self._g_proj = np.zeros(3)
        self._last_action = np.zeros(12)

        self._weights = {
            'base_v_xy': 4.0,
            'sigma_v_xy': 1.0,
            'base_v_z': -3.0,
            'angular_xy': -0.05,
            'yaw_rate': 1.0,
            'sigma_yaw': 0.25,
            'projected_gravity': -2.0,
            'effort': -1e-3,
            'joint_accel': -5e-7,
            'action_rate': -1e-2,
            'contact': -1.0,
            'feet_air_time': 2.0,
            'hip_q': -0.01,
            'thigh_q': -0.01
        }

        self._obs_weights = {
            'v': 2.0,
            'w': 0.25,
            'qd': 0.05
        }

        # observation stuff NOTE defined at end because of call to calc obs
        self._obs_clip_thresh = 100.0 # must be defined before _calc_obs
        self._single_obs_shape = self._calc_obs().shape
        self._obs_history = deque(
            [np.zeros(self._single_obs_shape) for _ in range(self._history_len)],
            maxlen=self._history_len
        )
        obs_shape = (self._single_obs_shape[0] * history_length,)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float64
        )

    def step(self, action: np.ndarray, populate_info=False):
        self._step += 1  # update for keeping time

        # calc action
        scaled_action = self._torque_scale * action
        clipped_action = np.clip(scaled_action, -1.0, 1.0)

        # step simulator and g_proj
        self.do_simulation(clipped_action, self.frame_skip)  
        self._g_proj = self.projected_gravity(self.data.qpos[3:7])

        # add observation to history
        obs = self._calc_obs()
        self._obs_history.append(obs)

        reward, reward_info = self._calc_reward(action)
        terminated, truncated = self._calc_term_trunc()
        if populate_info:
            info = {
                'speed': self.data.qvel[:3],
                'R_WBody': self.data.qpos[3:7],
                'distance_from_origin': np.linalg.norm(self.data.qpos[0:2], ord=2),
                'g_proj': self.projected_gravity(self.data.qpos[3:7]),
                **reward_info,
            }
        else:
            info = {}

        if self.render_mode == 'human' \
        and (self.data.time - self._last_render_time) > (1.0 / self.metadata['render_fps']):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action

        final_obs = np.concatenate(self._obs_history)

        return final_obs, reward, terminated, truncated, info

    def reset_model(self):
        # No noise on floating base
        noise = np.concatenate(([0] * 7, self._reset_rng.uniform(-0.05, 0.05, 12)))
        self.data.qpos[:] = self.model.key_qpos[0] + noise

        self.data.ctrl[:] = self.model.key_ctrl[0] + 0.1 * self.np_random.standard_normal(*self.data.ctrl.shape)
        
        # reset stateful things
        self._g_proj = self.projected_gravity(self.data.qpos[3:7])
        self._step = 0
        self._obs_history.extend([np.zeros(self._single_obs_shape) for _ in range(self._history_len)])
        self._last_render_time = -1.0
        self._last_action = np.zeros(12)
        self._last_contacts = np.zeros(4)
        self._feet_air_time = np.zeros(4)

        obs = self._calc_obs()
        self._obs_history.append(obs)

        return np.concatenate(self._obs_history)

    def _calc_term_trunc(self) -> tuple[bool, bool]:
        terminated = False

        body_quat = self.data.qpos[3:7]
        body_z_axis = self.R_from_quat(body_quat)[:, 2]
        cos_angle = np.dot(body_z_axis, self._upright)
        if cos_angle < 0.6:
            terminated = True  # Bad orientation

        body_z = self.data.qpos[2]
        if body_z < 0.2:
            terminated = True  # Fallen

        if not np.isfinite(self.state_vector()).all():
            terminated = True  # Something bad happened

        truncated = self._step >= (self._max_episode_time / self.dt)

        return terminated, truncated

    def _calc_obs(self):
        q = self.data.qpos[-12:]
        qd = self.data.qvel[-12:] * self._obs_weights['qd']
        v = self.data.qvel[:3] * self._obs_weights['v']
        w = self.data.qvel[3:6] * self._obs_weights['w']

        obs_unbounded = np.concatenate((q, v, w, qd, self._g_proj, self._last_action))
        obs_clipped = np.clip(obs_unbounded, -self._obs_clip_thresh, self._obs_clip_thresh)
        return obs_clipped
    
    def _calc_reward(self, action: np.ndarray) -> tuple[float, dict]:
        # https://arxiv.org/abs/2203.05194 ignore the delta t
        reward_info = {}

        # Base velocity xy
        d_v_xy = self._v_xy_desired - self.data.qvel[:2]
        reward_info['base_v_xy'] = self._weights['base_v_xy'] * np.exp(
            -np.square(d_v_xy).sum() / self._weights['sigma_v_xy']
        )

        # Base velocity z
        reward_info['base_v_z'] = self._weights['base_v_z'] * self.data.qvel[2] ** 2

        # Base angular velocity xy
        w_xy = self.data.qvel[3:5]
        reward_info['angular_xy'] = self._weights['angular_xy'] * np.square(w_xy).sum()

        # Base angular velocity z
        d_yaw = self._desired_yaw_rate - self.data.qvel[5]
        reward_info['yaw_rate'] = self._weights['yaw_rate'] * np.exp(
            -np.square(d_yaw).sum() / self._weights['sigma_yaw']
        )

        # Orientation
        # g_proj = self.projected_gravity(self.data.qpos[3:7])
        reward_info['projected_gravity'] = self._weights['projected_gravity'] * np.square(self._g_proj[:2]).sum()

        # Effort
        reward_info['effort'] = self._weights['effort'] * np.square(action).sum()

        # Joint accel
        qddot = self.data.qacc[-12:]
        reward_info['joint_accel'] = self._weights['joint_accel'] * np.square(qddot).sum()

        # Action rate
        d_action = self._last_action - action
        reward_info['action_rate'] = self._weights['action_rate'] * np.square(d_action).sum()

        # Hip Thigh Contact
        is_contact = (np.linalg.norm(self.data.cfrc_ext[self._contact_indices], axis=1) > 1e-3).astype(int)
        reward_info['contact'] = self._weights['contact'] * np.sum(is_contact)

        # Feet air time TODO check this again
        reward_info['feet_air_time'] = self._weights['feet_air_time'] * self.feet_air_time_reward

        #=======OPTIONAL==========
        # Hip position
        hip_q = self.data.qpos[self._hip_indices]
        d_hip_q = self._hip_q0 - hip_q
        reward_info['hip_q'] = self._weights['hip_q'] * np.abs(d_hip_q).sum()

        # Thigh position
        thigh_q = self.data.qpos[self._thigh_indices]
        d_thigh_q = self._thigh_q0 - thigh_q
        reward_info['thigh_q'] = self._weights['thigh_q'] * np.abs(d_thigh_q).sum()

        reward = 0.0
        for i in reward_info.values():
            reward += i
        reward += 0.1 # alive
        reward *= self.dt
        reward = max(reward, 0.0)
        return reward, reward_info
    
    @property
    def feet_air_time_reward(self) -> float:
        feet_contact_forces = self.data.cfrc_ext[self._feet_indices]
        feet_contact_forces = np.linalg.norm(feet_contact_forces, axis=1)

        feet_contact_force_mag = feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 0.5) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(self._v_xy_desired) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return air_time_reward

    @staticmethod
    def R_from_quat(quat) -> np.ndarray:
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def projected_gravity(self, quat) -> np.ndarray:
        '''
        Return the normalized gravity vector projected into the body frame using quaternion.
        this assumes gravity is pointing straight down
        '''
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        gx = 2 * (x * z - w * y) * self._gravity[2]
        gy = 2 * (y * z + w * x) * self._gravity[2]
        gz = (w * w - x * x - y * y + z * z) * self._gravity[2]

        vec = np.array([gx, gy, gz])
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm


if __name__ == '__main__':
    # =========INFO=========
    a1 = mujoco.MjModel.from_xml_path("./unitree_go1/scene_torque.xml")
    data = mujoco.MjData(a1)

    print("Number of joints:", a1.njnt)
    print("Number of actuators:", a1.nu)
    print("Number of bodies:", a1.nbody)
    print("Number of velocities:", a1.nv)

    print("Positions (qpos):", data.qpos)  # joint positions
    print("Velocities (qvel):", data.qvel)  # joint velocities
    print("Accelerations (qacc):", data.qacc)  # joint accelerations
    print("Actuator forces (ctrl):", data.ctrl)  # actuator commands

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

        print(
            f"Joint {i}: {name}, type={joint_type}, range={range_}, damping={damping}"
        )

    print(joint_lower_limits, "\n", joint_upper_limits)
    print(a1.jnt_limited)
    print(a1.opt.gravity)
    print(a1.actuator_ctrlrange)
    print(a1.key_qpos)

    # =============MINIMAL RENDER===================
    env = Go1_Env(history_length=2, render_mode='human')
    obs, info = env.reset()

    env.mujoco_renderer.render("human")

    for _ in range(2000):
        obs, reward, terminated, truncated, info = env.step(
            np.random.default_rng().uniform(-1, 1, 12)
        )
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    # =============BODY NAMES=======================
    # data = mujoco.MjData(a1)

    # mujoco.mj_step(a1, data)

    # body_names = [mujoco.mj_id2name(a1, mujoco.mjtObj.mjOBJ_BODY, i)
    #             for i in range(a1.nbody)]

    # for i in range(a1.ngeom):
    #     print(i, mujoco.mj_id2name(a1, mujoco.mjtObj.mjOBJ_GEOM, i),
    #         a1.geom_type[i], a1.geom_size[i])

    # for i, name in enumerate(body_names):
    #     cfrc = data.cfrc_ext[i]
    #     print(f"Body: {name}, CFRC: {cfrc}")
