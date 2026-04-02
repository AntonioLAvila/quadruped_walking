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
            impl='warp', # use mjx jax is basically unusable rip
            naconmax=4*8192,
            njmax=40,
            kick_config=config_dict.create(
                kick_wait_time=[0.05, 0.2], # s
                kick_vel=[0.0, 3.0],
                kick_duration=[0.05, 0.2] # s
            ),
            command_config=config_dict.create( # v_xy, yaw
                # Uniform distribution for command amplitude.
                bounds=[1.5, 0.8, 1.2],
                # Probability of not zeroing out new command.
                probs=[0.9, 0.25, 0.5]
            ),
            reward_config=config_dict.create(
                scales=config_dict.create(
                    # Tracking.
                    tracking_lin_vel=1.0,
                    tracking_ang_vel=0.5,
                    # Base reward.
                    lin_vel_z=-0.5,
                    ang_vel_xy=-0.05,
                    orientation=-5.0,
                    # Other.
                    dof_pos_limits=-1.0,
                    pose=0.5,
                    # Other.
                    termination=-1.0,
                    stand_still=-1.0,
                    # Regularization.
                    torques=-0.0002,
                    action_rate=-0.01,
                    energy=-0.001,
                    # Feet.
                    feet_clearance=-2.0,
                    feet_height=-0.2,
                    feet_slip=-0.1,
                    feet_air_time=0.1,
                ),
                tracking_sigma=0.25,
                max_foot_height=0.1,
            ),
            obs_config=config_dict.create(
                body_v=2.0,
                w=0.25,
                qd=0.05
            ),
            noise_config=config_dict.create(
                level=1.0,  # Set to 0.0 to disable noise.
                scales=config_dict.create(
                    q=0.03,
                    qd=1.5,
                    w=0.2,
                    gravity=0.05,
                    body_v=0.1,
                ),
            ),
        )
        super().__init__(cfg)

        self._xml_path = go2_mj_description.MJCF_PATH
        self._mj_model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self._q0 = jp.array(self.mjx_model.key_qpos.squeeze())

        self._ctrl_bounds = jp.array(self._config.command_config.bounds)
        self._ctrl_probs = jp.array(self._config.command_config.probs)

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
        q0 = self._q0.copy()
        v0 = jp.zeros(self.mjx_model.nv)

        # # xy +- 0.5
        # rng, key = jax.random.split(rng)
        # dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        # q0 = q0.at[0:2].add(dxy)

        # yaw in U(-pi, pi)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-jp.pi, maxval=jp.pi)
        quat = math.axis_angle_to_quat(jp.array([0,0,1]), yaw)
        q0 = q0.at[3:7].set(math.quat_mul(q0[3:7], quat))

        # small random noise on joints
        rng, key = jax.random.split(rng)
        noise = jax.random.uniform(key, (self.mjx_model.nu,), minval=-0.05, maxval=0.05)
        q0 = q0.at[-12:].add(noise)

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
        rng, key1, key2, key3 = jax.random.split(rng, 4)
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

        # command
        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_command = jax.random.exponential(key1) * 5 # 5 sec avg
        steps_until_command = jp.round(time_until_command / self.dt).astype(jp.int32)
        command = jax.random.uniform(
            key2,
            (3,),
            minval=-self._ctrl_bounds,
            maxval=self._ctrl_bounds
        )

        info = {
            "rng": rng,
            "command": command,
            "steps_until_next_cmd": steps_until_command,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "steps_until_kick": steps_until_kick,
            "kick_duration": kick_duration,
            "kick_steps": kick_steps,
            "steps_since_kick": 0,
            "kick_steps": 0,
            "kick_dir": jp.zeros(3),
            "kick_mag": kick_mag
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f'reward/{k}'] = jp.zeros(())
        metrics['swing_peak'] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)
        
    
    def step(self, state: State, action: jax.Array) -> State:
        pass

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        # extract obs
        q = data.qpos[-12:]
        qd = data.qvel[-12:]
        body_v = data.qvel[:3]
        w = data.qvel[3:6]
        gravity = math.rotate(self.mj_model.opt.gravity, data.qpos[3:7])

        # noise obs
        info['rng'], key1, key2, key3, key4, key5 = jax.random.split(info['rng'], 6)
        noisy_q = q + (2 * jax.random.uniform(key1, (12,)) - 1) * self._config.noise_config.level * self._config.noise_config.scales.q
        noisy_qd = qd + (2 * jax.random.uniform(key2, (12,)) - 1) * self._config.noise_config.level * self._config.noise_config.scales.qd
        noisy_body_v = body_v + \
            (2 * jax.random.uniform(key3, (3,)) - 1) * self._config.noise_config.level * self._config.noise_config.scales.body_v
        noisy_w = w + (2 * jax.random.uniform(key4, (3,)) - 1) * self._config.noise_config.level * self._config.noise_config.scales.w
        noisy_gravity = gravity + \
            (2 * jax.random.uniform(key5, (3,)) - 1) * self._config.noise_config.level * self._config.noise_config.scales.gravity

        state = jp.hstack([
            noisy_q - self._q0.copy()[-12:],
            noisy_qd,
            noisy_body_v,
            noisy_w,
            noisy_gravity,
            info['last_act'],
            info['command']
        ])

        privileged_state = jp.hstack([
            state,
            q - self._q0.copy()[-12:],
            qd,
            body_v,
            w,
            gravity,
            data.actuator_force,
            info['last_contact'],
            info['feet_air_time'],
            info['steps_since_kick'] >= info['steps_until_kick']
        ])

        return { # keywords for brax SAC and PPO with critic advantage
            'state': state,
            'privileged_state': privileged_state
        }
    
    def _get_termination(self, data:mjx.Data) -> jax.Array:
        body_quat = data.qpos[3:7]
        body_z_axis, _ = math.quat_to_axis_angle(body_quat)

        cos_angle = jp.dot(body_z_axis, [0,0,1])
        if cos_angle < 0.6:
            return True  # Bad orientation

        body_z = data.qpos[2]
        if body_z < 0.1:
            return True  # Fallen

        if not jp.isfinite(jp.concat([data.qpos, data.qvel])).all():
            return True  # Something bad happened

        return False
    


if __name__ == '__main__':
    # =========INFO=========
    # model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
    # print(go2_mj_description.MJCF_PATH)
    # data = mujoco.MjData(model)

    # print("Number of joints:", model.njnt)
    # print("Number of actuators:", model.nu)
    # print("Number of bodies:", model.nbody)
    # print("Number of velocities:", model.nv)
    # print("Number of sensors:", model.nsensor)


    # print("Positions (qpos):", data.qpos)  # joint positions
    # print("Velocities (qvel):", data.qvel)  # joint velocities
    # print("Accelerations (qacc):", data.qacc)  # joint accelerations
    # print("Actuator forces (ctrl):", data.ctrl)  # actuator commands

    # print('model time step', model.opt.timestep)

    # joint_upper_limits = []
    # joint_lower_limits = []
    # for i in range(model.njnt):
    #     name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    #     joint_type = model.jnt_type[i]  # 0=free, 1=ball, 2=slide, 3=hinge
    #     range_ = model.jnt_range[i]
    #     joint_lower_limits.append(float(range_[0]))
    #     joint_upper_limits.append(float(range_[1]))
    #     damping = model.dof_damping[i] if i < len(model.dof_damping) else None

    #     print(
    #         f"Joint {i}: {name}, type={joint_type}, range={range_}, damping={damping}"
    #     )
    
    # for i in range(model.nsensor):
    #     sensor_id = i
    #     sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
    #     sensor_type = model.sensor_type[i]
        
    #     # mjtSensor is an enum; this tells you if it's a touch sensor, gyro, etc.
    #     type_name = mujoco.mjtSensor(sensor_type).name
        
    #     print(f"ID: {sensor_id} | Name: {sensor_name} | Type: {type_name}")

    # # print('Joint limits:', joint_lower_limits, "\n", joint_upper_limits)
    # print('Limited joints:', model.jnt_limited)
    # print('Gravity:', model.opt.gravity)
    # print('Actuator ranges:', model.actuator_ctrlrange)
    # print('Key q_pos:', model.key_qpos)

    # =============MINIMAL RENDER===================
    env = Go2Env()
    rng = jax.random.PRNGKey(42)
    env.reset(rng)