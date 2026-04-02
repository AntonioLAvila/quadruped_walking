import jax
import jax.numpy as jp
from mujoco import mjx
import mujoco
from mujoco.mjx._src import math
from robot_descriptions import go2_mj_description
from mujoco_playground._src.mjx_env import MjxEnv, State, step, make_data
from ml_collections import config_dict
from typing import Mapping, Union, Dict, Any
Observation = Union[jax.Array, Mapping[str, jax.Array]]

class Go2Env(MjxEnv):
    def __init__(self):
        cfg = config_dict.create(
            ctrl_dt=0.002,
            sim_dt=0.002,
            episode_length=1000,
            action_scale=10,
            history_len=1,
            impl='mjx',
            naconmax=4*8192,
            njmax=40,
            kick_config=config_dict.create(
                kick_wait_time=[0.05, 0.2], # s
                kick_vel=[0, 3],
                kick_duration=[0.05, 0.2] # s
            )
        )
        super().__init__(cfg)

        self._xml_path = go2_mj_description.MJCF_PATH
        self._mj_model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
        self._mjx_model = mjx.put_model()
        self._prng_key = jax.random.PRNGKey(0)

    @property
    def xml_path(self):
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mj_model.nu
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
    
    def reset(self, rng: jax.Array) -> State:
        q0 = jp.array(self.mjx_model.key_qpos)
        v0 = jp.zeros(self.mjx_model.nv)

        # # xy +- 0.5
        # rng, key = jax.random.split(rng)
        # dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        # q0 = q0.at[0:2].add(dxy)

        # # yaw in U(-pi, pi)
        # rng, key = jax.random.split(rng)
        # yaw = jax.random.uniform(key, (1,), minval=-jp.pi, maxval=jp.pi)
        # quat = math.axis_angle_to_quat(jp.array([0,0,1]), yaw)
        # q0 = q0.at[3:7].set(math.quat_mul(q0[3:7], quat))

        # small random noise on joints
        rng, key = jax.random.split(rng)
        noise = jax.random.uniform(key, (self.mjx_model.nu,), minval=-0.05, maxval=0.05)
        q0 = q0.at[7:].add(noise)

        # random spatial body vel
        rng, key = jax.random.split(rng)
        v0 = v0.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        # random ctrl normal limited to 0.1
        rng, key = jax.random.split(rng)
        ctrl0 = jax.random.normal(key, (self.mjx_model.nu,)) * 0.1

        data = make_data(
            self.mj_model,
            qpos=q0,
            qvel=v0,
            ctrl=ctrl0,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax
        )
        data = mjx.forward(self.mjx_model, data)

        # kick
        rng, key1, key2, key3 = jax.random.split(key, 4)
        time_until_kick = jax.random.uniform(
            key1,
            minval=self._config.kick_config.kick_wait_time[0],
            maxval=self._config.kick_config.kick_wait_time[1]
        )
        steps_until_kick = jp.round(time_until_kick / self.dt).astype(jp.int32)
        kick_duration = jax.random.uniform(
            key2,
            minval=self._config.kick_config.kick_duration[0],
            maxval=self._config.kick_config.kick_duration[1]
        )
        kick_steps = jp.round(kick_duration / self.dt).astype(jp.int32)
        kick_mag = jax.random.uniform(
            key3,
            minval=self._config.kick_config.kick_vel[0],
            maxval=self._config.kick_config.kick_vel[1]
        )
        
    
    def step(self, state: State, action: jax.Array) -> State:
        pass

    def get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        pass
    
    


if __name__ == '__main__':
    # =========INFO=========
    model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)

    print("Number of joints:", model.njnt)
    print("Number of actuators:", model.nu)
    print("Number of bodies:", model.nbody)
    print("Number of velocities:", model.nv)

    print("Positions (qpos):", data.qpos)  # joint positions
    print("Velocities (qvel):", data.qvel)  # joint velocities
    print("Accelerations (qacc):", data.qacc)  # joint accelerations
    print("Actuator forces (ctrl):", data.ctrl)  # actuator commands

    print('model time step', model.opt.timestep)

    joint_upper_limits = []
    joint_lower_limits = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]  # 0=free, 1=ball, 2=slide, 3=hinge
        range_ = model.jnt_range[i]
        joint_lower_limits.append(float(range_[0]))
        joint_upper_limits.append(float(range_[1]))
        damping = model.dof_damping[i] if i < len(model.dof_damping) else None

        print(
            f"Joint {i}: {name}, type={joint_type}, range={range_}, damping={damping}"
        )

    # print('Joint limits:', joint_lower_limits, "\n", joint_upper_limits)
    print('Limited joints:', model.jnt_limited)
    print('Gravity:', model.opt.gravity)
    print('Actuator ranges:', model.actuator_ctrlrange)
    print('Key q_pos:', model.key_qpos)

    print(type(model))

    # =============MINIMAL RENDER===================
    # env = Go2Env()