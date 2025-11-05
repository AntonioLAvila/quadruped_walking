'''
I told gemini to rewrite ts
'''
from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from typing import Callable, Tuple
import flax

# NOTE: Assuming 'util' and 'robot_descriptions' are available.
from util import (
    R_from_quat,
    projected_gravity,
    weights,
)
from robot_descriptions import a1_mj_description

# --- Custom JAX-compatible State definition (Same as before) ---

@flax.struct.dataclass
class A1EnvState(State):
    """
    Extends Brax's base.State to hold custom environment data.
    The order of fields is corrected to place non-default (custom) fields first.
    """
    # --- Custom Fields (No default values) ---
    step: jp.ndarray  
    last_action: jp.ndarray
    last_contacts: jp.ndarray
    feet_air_time: jp.ndarray
    obs_buffer: jp.ndarray
    obs_buffer_idx: jp.ndarray
    rng: jax.random.PRNGKey

    # --- Inherited Fields (Must follow custom fields) ---
    # These fields must be defined in the same order as the base class
    pipeline_state: base.State  
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: dict
    info: dict

# --- Corrected Observation Function (JAX pure function) ---

def simple_obs(pipeline_state: base.State) -> jax.Array:
    """
    Extracts the corrected observation:
    [quat(4), joint_angles(12), all_velocities(18), projected_gravity(3)]
    Total size: 4 + 12 + 18 + 3 = 37
    """
    # q = [pos(3), quat(4), joint_angles(12)] (total 19)
    # qd = [lin_vel(3), ang_vel(3), joint_vel(12)] (total 18)
    
    # 1. Base Orientation (quat: q[3:7])
    q_orient = pipeline_state.q[3:7] 
    
    # 2. Joint Positions (q[7:19])
    q_joint = pipeline_state.q[7:] 
    
    # 3. All Velocities (qd[0:18])
    v_obs = pipeline_state.qd 
    
    # 4. Projected Gravity
    g_proj = projected_gravity(q_orient)
    
    return jp.concatenate((q_orient, q_joint, v_obs, g_proj))

# --- JAX-compatible Ring Buffer Functions (Same as before) ---

def _make_ring_buffer(buffer_size: int, obs_shape: Tuple[int, ...]) -> Tuple[jp.ndarray, jp.ndarray]:
    buffer = jp.zeros((buffer_size,) + obs_shape)
    index = jp.array(0, dtype=jp.int32)
    return buffer, index

def _update_ring_buffer(buffer: jp.ndarray, index: jp.ndarray, new_obs: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
    buffer = buffer.at[index].set(new_obs)
    index = (index + 1) % buffer.shape[0]
    return buffer, index

def _get_recent(buffer: jp.ndarray, index: jp.ndarray, n: int) -> jp.ndarray:
    buffer_size = buffer.shape[0]
    idxs = (index - n + buffer_size) % buffer_size
    return buffer[idxs]


# --- Environment Class (Updates to _OBS_SIZE and the reset method) ---

class A1_Env(PipelineEnv):

    # Corrected default observation size: 4 + 12 + 18 + 3 = 37
    _OBS_SIZE = 37 

    def __init__(
        self,
        get_obs: Callable[[base.State], jax.Array] = simple_obs,
        history_len=1,
        torque_scale=1.0,
        rng_seed=0,
        **kwargs
    ):
        sys = mjcf.load(a1_mj_description.MJCF_PATH)
        
        super().__init__(
            sys=sys,
            backend='mjx',
            n_frames=5, # dt_control = 0.01
            **kwargs,
        )
        
        self._get_obs = get_obs
        self._rng = jax.random.PRNGKey(rng_seed)
        self._max_episode_time = 15.0 # seconds
        self._torque_scale = torque_scale
        # Floating base (3 pos + 4 quat) + 12 joints
        self._q0 = jp.array([0., 0., 0.3] + [1., 0., 0., 0.] + [0., jp.pi/4, -jp.pi/2]*4)
        self._torque_limit = 33.5
        self._history_len = history_len

        # Used in reward (as constant class properties)
        self._upright = jp.array([0., 0., 1.])
        self._v_xy_desired = jp.array([1., 0.])
        self._desired_yaw_rate = 0.0
        self._contact_indices = jp.array([2, 3, 5, 6, 8, 9, 11, 12])
        self._feet_indices = jp.array([4, 7, 10, 13])
        self._hip_indices = jp.array([7, 10, 13, 16])
        self._thigh_indices = jp.array([8, 11, 14, 17])
        self._hip_q0 = jp.zeros(4)
        self._thigh_q0 = jp.ones(4) * (jp.pi/4)

    def reset(self, rng: jax.Array) -> A1EnvState:
        """Resets the environment and returns the initial state."""
        
        rng, noise_rng, state_rng = jax.random.split(rng, 3)
        
        # Base state: qpos/qd noise
        # Only apply noise to joints, not floating base (first 7 DOFs)
        noise_qpos = jp.concatenate(([0.]*7, jax.random.uniform(noise_rng, (12,), minval=-0.05, maxval=0.05)))
        qpos = self._q0 + noise_qpos
        qvel = jp.zeros(self.sys.qd_size())
        
        pipeline_state = self.pipeline_init(qpos, qvel)

        action_rng, _ = jax.random.split(state_rng)
        init_action = jax.random.normal(action_rng, (12,)) * 0.1
        
        current_obs = self._get_obs(pipeline_state)
        
        # Initialize ring buffer
        obs_shape = current_obs.shape
        buffer_size = self._history_len
        obs_buffer, obs_buffer_idx = _make_ring_buffer(buffer_size, obs_shape)
        
        # Add initial observation to the buffer
        obs_buffer, obs_buffer_idx = _update_ring_buffer(obs_buffer, obs_buffer_idx, current_obs)

        # Final observation includes history
        final_obs = _get_recent(obs_buffer, obs_buffer_idx, self._history_len).flatten()

        # Create custom A1EnvState
        state = A1EnvState(
            pipeline_state=pipeline_state,
            obs=final_obs,
            reward=jp.zeros(()),
            done=jp.zeros((), dtype=jp.bool_),
            metrics={},
            info={
                'truncation': jp.zeros((), dtype=jp.bool_),
                'reward_details': {}
            },
            # Custom fields
            step=jp.array(0, dtype=jp.int32),
            last_action=init_action,
            last_contacts=jp.zeros(4),
            feet_air_time=jp.zeros(4),
            obs_buffer=obs_buffer,
            obs_buffer_idx=obs_buffer_idx,
            rng=rng,
        )
        return state

    # Step, _calc_reward, _get_term_trunc, and _feet_air_time_reward methods are functionally the same as the last version, 
    # as they correctly use pure functions and the State object.
    
    def step(self, state: A1EnvState, action: jax.Array) -> A1EnvState:
        """Performs one environment step and returns the next state."""
        
        scaled_action = action * self._torque_scale
        ctrl = jp.clip(scaled_action, -self._torque_limit, self._torque_limit)

        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        new_step = state.step + 1
        terminated, truncated = self._get_term_trunc(pipeline_state, new_step)
        done = terminated | truncated

        reward, reward_info, new_feet_air_time, new_last_contacts = self._calc_reward(
            pipeline_state, ctrl, state.last_action, state.feet_air_time, state.last_contacts
        )
        
        current_obs = self._get_obs(pipeline_state)
        new_obs_buffer, new_obs_buffer_idx = _update_ring_buffer(state.obs_buffer, state.obs_buffer_idx, current_obs)
        final_obs = _get_recent(new_obs_buffer, new_obs_buffer_idx, self._history_len).flatten()

        return state.replace(
            pipeline_state=pipeline_state,
            obs=final_obs,
            reward=reward,
            done=done,
            metrics=state.metrics.copy(),
            info={
                'truncation': truncated,
                'reward_details': reward_info
            },
            step=new_step,
            last_action=ctrl,
            last_contacts=new_last_contacts,
            feet_air_time=new_feet_air_time,
            obs_buffer=new_obs_buffer,
            obs_buffer_idx=new_obs_buffer_idx,
        )
        
    def _calc_reward(
        self,
        pipeline_state: base.State,
        action: jp.ndarray,
        last_action: jp.ndarray,
        feet_air_time: jp.ndarray,
        last_contacts: jp.ndarray,
    ) -> Tuple[jp.ndarray, dict, jp.ndarray, jp.ndarray]:
        """Calculates the reward components. Pure function."""
        
        qpos = pipeline_state.q
        qvel = pipeline_state.qd
        cfrc_ext = pipeline_state.cfrc_ext
        qacc = pipeline_state.qdd
        
        reward_info = {}

        # Base velocity xy
        v_xy = qvel[:2]
        d_v_xy = self._v_xy_desired - v_xy
        reward_info['base_v_xy'] = weights['base_v_xy'] * jp.exp(-jp.square(d_v_xy).sum() / weights['sigma_v_xy'])

        # Base velocity z
        v_z = qvel[2]
        reward_info['base_v_z'] = weights['base_v_z'] * jp.square(v_z)

        # Base angular velocity xy
        w_xy = qvel[3:5]
        reward_info['angular_xy'] = weights['angular_xy'] * jp.square(w_xy).sum()

        # Base angular velocity z
        w_z = qvel[5]
        d_yaw = self._desired_yaw_rate - w_z
        reward_info['yaw_rate'] = weights['yaw_rate'] * jp.exp(-jp.square(d_yaw).sum() / weights['sigma_yaw'])

        # Orientation
        quat = qpos[3:7]
        reward_info['projected_gravity'] = weights['projected_gravity'] * jp.square(projected_gravity(quat)[:2]).sum()

        # Effort
        reward_info['effort'] = weights['effort'] * jp.square(action).sum()

        # Joint accel
        qddot = qacc[-12:]
        reward_info['joint_accel'] = weights['joint_accel'] * jp.square(qddot).sum()

        # Action rate
        d_action = last_action - action
        reward_info['action_rate'] = weights['action_rate'] * jp.square(d_action).sum()

        # Hip Thigh Contact
        is_contact = (jp.linalg.norm(cfrc_ext[self._contact_indices], axis=1) > 1e-3).astype(jp.float32)
        reward_info['contact_penalty'] = weights['contact'] * jp.sum(is_contact)

        # Feet air time reward and update
        air_time_reward, new_feet_air_time, new_last_contacts = self._feet_air_time_reward(
            cfrc_ext, feet_air_time, last_contacts
        )
        reward_info['feet_air_time'] = weights['feet_air_time'] * air_time_reward
        
        # Hip position
        hip_q = qpos[self._hip_indices]
        d_hip_q = self._hip_q0 - hip_q
        reward_info['hip_q'] = weights['hip_q'] * jp.abs(d_hip_q).sum()

        # Thigh position
        thigh_q = qpos[self._thigh_indices]
        d_thigh_q = self._thigh_q0 - thigh_q
        reward_info['thigh_q'] = weights['thigh_q'] * jp.abs(d_thigh_q).sum()

        reward = jp.sum(jp.array(list(reward_info.values())))
 
        return reward, reward_info, new_feet_air_time, new_last_contacts

    def _get_term_trunc(self, pipeline_state: base.State, step: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Calculates termination and truncation conditions. Pure function."""
        
        qpos = pipeline_state.q
        
        body_quat = qpos[3:7]
        body_z_axis = R_from_quat(body_quat)[:, 2]
        cos_angle = jp.dot(body_z_axis, self._upright)
        bad_orientation = cos_angle < 0.25

        body_z = qpos[2]
        fallen = body_z < 0.1

        not_finite = ~jp.all(jp.isfinite(qpos)) | ~jp.all(jp.isfinite(pipeline_state.qd))

        terminated = bad_orientation | fallen | not_finite
        
        truncated = step >= (self._max_episode_time / self.dt)

        return terminated, truncated

    def _feet_air_time_reward(
        self,
        cfrc_ext: jp.ndarray,
        feet_air_time: jp.ndarray,
        last_contacts: jp.ndarray,
    ) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
        """Calculates air time reward and updates air time and contact state. Pure function."""

        feet_contact_forces = cfrc_ext[self._feet_indices]
        feet_contact_force_mag = jp.linalg.norm(feet_contact_forces, axis=1)
        curr_contact = feet_contact_force_mag > 1.0

        contact_filter = jp.logical_or(curr_contact, last_contacts)

        new_last_contacts = curr_contact

        new_feet_air_time = feet_air_time + self.dt * (~curr_contact)

        first_contact = (feet_air_time > 0.0) & contact_filter

        air_time_reward = jp.sum((feet_air_time - 0.1) * first_contact)
        
        is_moving = jp.linalg.norm(self._v_xy_desired) > 0.1
        air_time_reward = jp.where(is_moving, air_time_reward, jp.zeros_like(air_time_reward))

        new_feet_air_time = jp.where(contact_filter, 0.0, new_feet_air_time)

        return air_time_reward, new_feet_air_time, new_last_contacts
    

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    env = A1_Env()
    action_size = env.action_size # Should be 12

    # 1. JIT Reset Test
    jitted_reset = jax.jit(env.reset)
    key, reset_key = jax.random.split(key)
    state = jitted_reset(reset_key)
    print("JIT Reset successful.")
    print(f"Initial State Obs shape: {state.obs.shape}")

    # 2. JIT Step Test
    jitted_step = jax.jit(env.step)
    dummy_action = jp.zeros(action_size)
    state = jitted_step(state, dummy_action)
    print("JIT Step successful.")
    print(f"Obs after step shape: {state.obs.shape}")