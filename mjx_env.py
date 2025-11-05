from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco
from robot_descriptions import a1_mj_description
from collections import deque
from typing import Callable, Tuple
from copy import deepcopy
from util import (
    A1_joint_lb,
    A1_joint_ub,
    dt_control,
    dt_sim,
    default_cam_config,
    control_freq,
    R_from_quat,
    projected_gravity,
    weights,
)


def simple_obs(pipeline_state: base.State) -> jax.Array:
    q_obs = pipeline_state.q[:3]
    v_obs = pipeline_state.qd
    g_proj = projected_gravity(q_obs[:4])
    return jp.concatenate((q_obs, v_obs, g_proj))


class A1_Env(PipelineEnv):

    def __init__(
        self,
        get_obs: Callable[[State], jax.Array],
        history_len=0,
        torque_scale=1,
        **kwargs
    ):
        sys = mjcf.load(a1_mj_description.MJCF_PATH)
        super(). __init__(
            self,
            sys=sys,
            backend='mjx',
            n_frames=5, # NOTE this give 5 = 0.01/0.002 -> dt_control = 0.01
        )
        self._get_obs = get_obs
        self._reset_rng = jp.random.default_rng()
        self._max_episode_time = 15.0 # seconds
        self._torque_scale = torque_scale
        self._q0 = jp.array([0, 0, 0.3] + [1, 0, 0, 0] + [0, jp.pi/4, -jp.pi/2]*4)
        self._torque_limit = 33.5

        # Used in reward
        self._upright = jp.array([0,0,1])
        self._v_xy_desired = jp.array([1,0])
        self._desired_yaw_rate = 0.0
        self._contact_indices = [2, 3, 5, 6, 8, 9, 11, 12] # hip and thigh
        self._hip_indices = [7, 10, 13, 16]
        self._thigh_indices = [8, 11, 14, 17]
        self._hip_q0 = jp.zeros(4)
        self._thigh_q0 = jp.ones(4) * (jp.pi/4)
        self._feet_indices = [4, 7, 10, 13]

        # Stateful things
        self._step = 0
        self._last_action = jp.zeros(12) # NOTE this is seperate from _data_history because it's used only for reward calculation
        self._last_contacts = jp.zeros(4)
        self._feet_air_time = jp.zeros(4)
        
    def step(self, state: State, action: jax.Array) -> State:
        self._step += 1 # update for keeping time

        # calc scaled torque
        action = action*self._torque_scale
        action = jp.clip(action, -self._torque_limit, self._torque_limit)

        # step physics
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        obs = self.observation_extractor.calc_obs(self.data, self._data_history)
        reward, reward_info = self._calc_reward(action)
        terminated, truncated = self.is_term_trunc
        info = {
            'p_WBody': self.data.qpos[:3],
            'R_WBody': self.data.qpos[3:7],
            'g_proj': projected_gravity(self.data.qpos[3:7]),
            **reward_info
        }
        
        self._last_action = action.copy()

        return obs, reward, terminated, truncated, info
    
    def reset(self) -> State:
        # No noise on floating base
        noise = jp.concatenate(([0]*7, self._reset_rng.uniform(-0.05, 0.05, 12)))
        self.data.qpos[:] = self._q0 + noise

        self.data.ctrl[:] = self._reset_rng.normal(0, 0.1, 12)

        self._step = 0
        self._data_history.clear()
        self._last_render_time = -1.0
        self._last_action = jp.zeros(12)
        self._last_contacts = jp.zeros(4)
        self._feet_air_time = jp.zeros(4)

        observation = self.observation_extractor.calc_obs(self.data, self._data_history)
        return observation
        
    def _calc_reward(self, action: jp.ndarray) -> tuple[float, dict]:
        # https://arxiv.org/abs/2203.05194 ignore the delta t
        reward_info = {}

        # Base velocity xy
        v_xy = self.data.qvel[:2]
        d_v_xy = self._v_xy_desired - v_xy
        reward_info['base_v_xy'] = weights['base_v_xy'] * jp.exp(-jp.square(d_v_xy).sum()/weights['sigma_v_xy'])

        # Base velocity z
        v_z = self.data.qvel[2]
        reward_info['base_v_z'] = weights['base_v_z'] * v_z**2

        # Base angular velocity xy
        w_xy = self.data.qvel[3:5]
        reward_info['angular_xy'] = weights['angular_xy'] * jp.square(w_xy).sum()

        # Base angular velocity z
        w_z = self.data.qvel[5]
        d_yaw = self._desired_yaw_rate - w_z
        reward_info['yaw_rate'] = weights['yaw_rate'] * jp.exp(-jp.square(d_yaw).sum()/weights['sigma_yaw'])

        # Orientation
        quat = self.data.qpos[3:7]
        reward_info['projected_gravity'] = weights['projected_gravity'] * jp.square(projected_gravity(quat)[:2]).sum()

        # Effort
        reward_info['effort'] = weights['effort'] * jp.square(action).sum()

        # Joint accel
        qddot = self.data.qacc[-12:]
        reward_info['joint_accel'] = weights['joint_accel'] * jp.square(qddot).sum()

        # Action rate
        d_action = self._last_action - action
        reward_info['action_rate'] = weights['action_rate'] * jp.square(d_action).sum()

        # Hip Thigh Contact
        is_contact = (jp.linalg.norm(self.data.cfrc_ext[self._contact_indices], axis=1) > 1e-3).astype(int)
        reward_info['contact'] = weights['contact'] * jp.sum(is_contact)

        # Feet air time TODO check this again
        reward_info['feet_air_time'] = weights['feet_air_time'] * self.feet_air_time_reward

        #=======OPTIONAL==========
        # Hip position
        hip_q = self.data.qpos[self._hip_indices]
        d_hip_q = self._hip_q0 - hip_q
        reward_info['hip_q'] = weights['hip_q'] * jp.abs(d_hip_q).sum()

        # Thigh position
        thigh_q = self.data.qpos[self._thigh_indices]
        d_thigh_q = self._thigh_q0 - thigh_q
        reward_info['thigh_q'] = weights['thigh_q'] * jp.abs(d_thigh_q).sum()

        # Alive
        # reward_info['alive'] = 1000

        reward = 0.0
        for i in reward_info.values():
            reward += i
 
        return reward, reward_info
    
    @property
    def is_term_trunc(self) -> tuple[bool, bool]:
        terminated = False
        
        body_quat = self.data.qpos[3:7]
        body_z_axis = R_from_quat(body_quat)[:, 2]
        cos_angle = jp.dot(body_z_axis, self._upright)
        if cos_angle < 0.25:
            terminated = True # Bad orientation

        body_z = self.data.qpos[2]
        if body_z < 0.1:
            terminated = True # Fallen

        if not jp.isfinite(self.state_vector()).all():
            terminated = True # Something bad happened

        truncated = self._step >= (self._max_episode_time / self.dt)

        return terminated, truncated

    @property
    def feet_air_time_reward(self) -> float:
        feet_contact_forces = self.data.cfrc_ext[self._feet_indices]
        feet_contact_forces = jp.linalg.norm(feet_contact_forces, axis=1)

        feet_contact_force_mag = feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = jp.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = jp.sum((self._feet_air_time - 0.1) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= jp.linalg.norm(self._v_xy_desired) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return air_time_reward
    
    def make_ring_buffer(buffer_size: int, obs_shape: Tuple[int, ...]) -> Tuple[jp.ndarray, jp.ndarray]:
        buffer = jp.zeros((buffer_size,) + obs_shape)
        index = jp.array(0, dtype=jp.int32)
        return buffer, index

    def update_ring_buffer(buffer: jp.ndarray, index: jp.ndarray, new_obs: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        buffer = buffer.at[index % buffer.shape[0]].set(new_obs)
        index = (index + 1) % buffer.shape[0]
        return buffer, index

    def get_recent(buffer: jp.ndarray, index: jp.ndarray, n: int) -> jp.ndarray:
        idxs = (index - n + jp.arange(n)) % buffer.shape[0]
        return buffer[idxs]

    def clear_ring_buffer(buffer: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        cleared_buffer = jp.zeros_like(buffer)
        reset_index = jp.array(0, dtype=jp.int32)
        return cleared_buffer, reset_index

