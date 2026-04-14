import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
import mujoco
from mujoco.mjx._src import math
from mujoco_playground._src.mjx_env import MjxEnv, State, step, make_data, update_assets, get_sensor_data
from ml_collections import config_dict
from typing import Any
import mediapy as media

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
            njmax=156,
            soft_joint_limit_factor=0.9,
            kick_config=config_dict.create(
                kick_wait_time=[0.05, 0.2], # s
                kick_vel=[0.0, 3.0],
                kick_duration=[0.05, 0.2], # s
                enable=False
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
            noise_config=config_dict.create(
                level=1.0,  # Set to 0.0 to disable noise.
                scales=config_dict.create(
                    q=0.03,
                    qd=1.5,
                    gyro=0.2,
                    gravity=0.05,
                    body_v=0.1
                ),
            ),
        )
        super().__init__(cfg)

        # load model
        self._xml_path = 'unitree_go2/scene_warp.xml'
        assets = update_assets({}, 'unitree_go2/assets')
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path, assets=assets)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # turn things into jax arrays
        self._q0 = jp.array(self.mjx_model.key_qpos.squeeze())
        self._ctrl_bounds = jp.array(self._config.command_config.bounds)
        self._ctrl_probs = jp.array(self._config.command_config.probs)
        lowers, uppers = self.mj_model.jnt_range[1:].T # NOTE first joint is the free body
        self._soft_lowers = lowers * self._config.soft_joint_limit_factor
        self._soft_uppers = uppers * self._config.soft_joint_limit_factor

        # sites
        FEET = ['FL', 'RL', 'FR', 'RR']
        self._feet_site_ids = np.array([self._mj_model.site(f'{name}_site').id for name in FEET])
        self._imu_site_id = self._mj_model.site('imu').id
        self._torso_body_id = self.mj_model.body('base').id

        # sensors
        self._feet_floor_sensor_ids = np.array([self._mj_model.sensor(f'{name}_floor_contact').id for name in FEET])
        self._global_angvel_sensor_name = 'global_angvel'
        self._accel_sensor_name = 'accelerometer'
        self._local_linvel_sensor_name = 'local_linvel'
        self._global_linvel_sensor_name = 'global_linvel'
        self._gyro_sensor_name = 'gyro'
        self._body_z_axis_sensor_name = 'body_z_axis'
        self._feet_linvel_sensor_names = np.array([f'{name}_vel' for name in FEET])

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
            "steps_until_cmd": steps_until_command,
            "last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "kick_duration": kick_duration,
            "kick_steps": kick_steps,
            "steps_since_kick": 0,
            "steps_until_kick": steps_until_kick,
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
        # maybe kick
        if self._config.kick_config.enable:
            state = self._handle_kick(state)

        # step
        scaled_action = action*self._config.action_scale
        data = step(self.mjx_model, state.data, scaled_action, self.n_substeps)

        # handle feet movement
        contact = jp.array([
            data.sensordata[self.mj_model.sensor_adr[sid]] > 0
            for sid in self._feet_floor_sensor_ids
        ])
        filter = contact | state.info['last_contact']
        first_contact = (state.info["feet_air_time"] > 0) * filter
        state.info["feet_air_time"] += self.dt
        p_foot = data.site_xpos[self._feet_site_ids]
        p_foot_z = p_foot[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_foot_z)

        # observe
        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        # rewards
        rewards = self._get_reward(data, action, state.info, done, first_contact, contact)
        # NOTE rewards config needs to line up with the dict returned by _get_reward
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()} 
        final_reward = jp.maximum(sum(rewards.values()) * self.dt, 0.0)

        # update info
        state.info['last_act'] = action
        state.info['steps_until_cmd'] -= 1
        state.info['feet_air_time'] *= ~contact
        state.info['last_contact'] = contact
        state.info['swing_peak'] = ~contact
        state.info['rng'], key1, key2 = jax.random.split(state.info['rng'], 3)
        state.info['command'] = jp.where(
            state.info['steps_until_cmd'] <= 0,
            self._sample_command(key1, state.info['command']),
            state.info['command']
        )
        state.info['steps_until_cmd'] = jp.where(
            done | (state.info['steps_until_cmd'] <= 0),
            jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
            state.info['steps_until_cmd']
        )

        # update metrics
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(final_reward.dtype)
        state = state.replace(data=data, obs=obs, reward=final_reward, done=done)
        return state


    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        
        # extract obs
        q = data.qpos[-12:]
        qd = data.qvel[-12:]
        body_v = get_sensor_data(self.mj_model, data, self._local_linvel_sensor_name)
        gyro = get_sensor_data(self.mj_model, data, self._gyro_sensor_name)
        gravity = data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

        # noise obs
        info['rng'], key1, key2, key3, key4, key5 = jax.random.split(info['rng'], 6)
        noisy_q = q + (2 * jax.random.uniform(key1, q.shape) - 1) * self._config.noise_config.level * self._config.noise_config.scales.q
        noisy_qd = qd + (2 * jax.random.uniform(key2, qd.shape) - 1) * self._config.noise_config.level * self._config.noise_config.scales.qd
        noisy_body_v = body_v + \
            (2 * jax.random.uniform(key3, body_v.shape) - 1) * self._config.noise_config.level * self._config.noise_config.scales.body_v
        noisy_gyro = gyro + (2 * jax.random.uniform(key4, gyro.shape) - 1) * self._config.noise_config.level * self._config.noise_config.scales.gyro
        noisy_gravity = gravity + \
            (2 * jax.random.uniform(key5, gravity.shape) - 1) * self._config.noise_config.level * self._config.noise_config.scales.gravity

        state = jp.hstack([
            noisy_q - self._q0[-12:],
            noisy_qd,
            noisy_body_v,
            noisy_gyro,
            noisy_gravity,
            info['last_act'],
            info['command']
        ])

        accelerometer = get_sensor_data(self.mj_model, data, self._accel_sensor_name)
        angvel = get_sensor_data(self.mj_model, data, self._global_angvel_sensor_name)
        feet_vel = self._get_feet_vel(data)

        privileged_state = jp.hstack([
            state,
            q - self._q0[-12:],
            qd,
            body_v,
            gyro,
            gravity,
            accelerometer,
            angvel,
            feet_vel.flatten(),
            data.actuator_force,
            data.xfrc_applied[self._torso_body_id, :3],
            info['last_contact'],
            info['feet_air_time'],
            info['steps_since_kick'] >= info['steps_until_kick']
        ])

        return { # keywords for brax SAC and PPO with critic advantage
            'state': state,
            'privileged_state': privileged_state
        }
    

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        is_finite = jp.isfinite(jp.concat([data.qpos, data.qvel])).all()

        terminated = (get_sensor_data(self.mj_model, data, self._body_z_axis_sensor_name)[-1] < 0.0) | (~is_finite)

        return terminated
    
    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, jax.Array],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array
    ) -> dict[str, jax.Array]:
        # return {k: jp.ones(1) for k in self._config.reward_config.scales.keys()}
        return {
            "tracking_lin_vel": self._linvel_tracking(
                info["command"],
                get_sensor_data(self.mj_model, data, self._local_linvel_sensor_name)
            ),
            "tracking_ang_vel": self._angvel_tracking(
                info["command"],
                get_sensor_data(self.mj_model, data, self._gyro_sensor_name)
            ),
            "lin_vel_z": self._cost_linvel_z(get_sensor_data(self.mj_model, data, self._global_linvel_sensor_name)),
            "ang_vel_xy": self._cost_angvel_xy(get_sensor_data(self.mj_model, data, self._global_angvel_sensor_name)),
            "orientation": self._cost_orientation(get_sensor_data(self.mj_model, data, self._body_z_axis_sensor_name)),
            "stand_still": self._inaction_cost(info["command"], data.qpos[-12:]),
            "termination": self._cost_termination(done),
            "pose": self._pose_reward(data.qpos[-12:]),
            "torques": self._torque_cost(action),
            "action_rate": self._action_rate_cost(
                action,
                info["last_act"]
            ),
            "energy": self._energy_cost(data.qvel[-12:], data.actuator_force),
            "feet_slip": self._feet_slip_cost(data, contact, info),
            "feet_clearance": self._feet_clearance_cost(data),
            "feet_height": self._feet_height_cost(
                info["swing_peak"],
                first_contact,
                info
            ),
            "feet_air_time": self._feet_air_time_reward(
                info["command"],
                first_contact,
                info["feet_air_time"]
            ),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[-12:]),
        }
    
    def _handle_kick(self, orig_state: State) -> State:
        def kick(state: State) -> State:
            # TODO implement
            return state

        def dont_kick(state: State) -> State:
            # TODO implement
            return state

        return jax.lax.cond(
            orig_state.info['steps_since_kick'] >= orig_state.info['steps_until_kick'],
            kick,
            dont_kick,
            orig_state
        )
    
    def _sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
        # command sampling for robustness just copied
        # it basically jitters the command a little
        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        y_k = jax.random.uniform(
            y_rng, shape=(3,), minval=-self._ctrl_bounds, maxval=self._ctrl_bounds
        )
        z_k = jax.random.bernoulli(z_rng, self._ctrl_probs, shape=(3,))
        w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
        x_kp1 = x_k - w_k * (x_k - y_k * z_k)
        return x_kp1
    
    def _get_feet_vel(self, data: mjx.Data):
        return jp.stack([
            get_sensor_data(self.mj_model, data, name)
            for name in self._feet_linvel_sensor_names
        ])
    
    # ========== REWARDS ==============
    def _linvel_tracking(self, commands: jax.Array, local_linvel: jax.Array) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_linvel[:2]))
        return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)
    
    def _angvel_tracking(self, commands: jax.Array, local_angvel: jax.Array) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - local_angvel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)
    
    def _cost_linvel_z(self, global_linvel: jax.Array) -> jax.Array:
        return jp.square(global_linvel[2])
    
    def _cost_angvel_xy(self, global_angvel) -> jax.Array:
        return jp.sum(jp.square(global_angvel[:2]))
    
    def _cost_orientation(self, body_z_axis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(body_z_axis[:2]))
    
    def _torque_cost(self, act: jax.Array) -> jax.Array:
        return jp.sqrt(jp.sum(jp.square(act))) + jp.sum(jp.abs(act))
    
    def _energy_cost(self, qd: jax.Array, qfrc_actuator) -> jax.Array:
        return jp.sum(jp.abs(qd) * jp.abs(qfrc_actuator))
    
    def _action_rate_cost(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        return jp.sum(jp.square(act - last_act))
    
    def _pose_reward(self, q: jax.Array) -> jax.Array:
        return jp.exp(-jp.sum(jp.square(q - self._q0[-12:])))
    
    def _inaction_cost(self, commands: jax.Array, q: jax.Array) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(q - self._q0[-12:])) * (cmd_norm < 0.01)
    
    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done
    
    def _cost_joint_pos_limits(self, q: jax.Array) -> jax.Array:
        out_of_limits = -jp.clip(q - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(q - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)
    
    def _feet_slip_cost(self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = self._get_feet_vel(data)
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

    def _feet_clearance_cost(self, data: mjx.Data) -> jax.Array:
        feet_vel = self._get_feet_vel(data)
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_ids]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)
    
    def _feet_height_cost(self, swing_peak: jax.Array, first_contact: jax.Array, info: dict[str, Any]) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

    def _feet_air_time_reward(self, commands: jax.Array, first_contact: jax.Array, air_time: jax.Array) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
        return rew_air_time




if __name__ == '__main__':
    # env = Go2Env()
    # rng = jax.random.PRNGKey(42)
    # s = env.reset(rng)
    # for _ in range(10):
    #     s = env.step(s, jp.zeros(12))
    # print('done')

    env = Go2Env()
    rng = jax.random.PRNGKey(42)
    
    # 1. Reset the environment
    state = env.reset(rng)
    
    # 2. Rollout the environment and store states
    trajectory = []
    for _ in range(100):  # Run for more steps to see meaningful movement
        trajectory.append(state)
        # Use a random action or zeros
        action = jp.zeros(env.action_size) 
        state = env.step(state, action)
    
    print('Simulation complete. Rendering...')

    # 3. Render the trajectory
    # This returns a list of numpy arrays (RGB frames)
    frames = env.render(trajectory, camera='track') # 'track' is common for quadrupeds

    # 4. Save or Show the video
    media.write_video('go2_simulation.mp4', frames, fps=1.0/env.dt)
    print('Video saved to go2_simulation.mp4')