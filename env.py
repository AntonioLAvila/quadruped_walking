import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco
from brax.envs.base import PipelineEnv, State
from robot_descriptions import go2_mj_description
from brax.io import mjcf


class Go2Env(PipelineEnv):
    def __init__(self):
        sys = mjcf.load(go2_mj_description.MJCF_PATH)

        super().__init__(sys, backend='mjx', n_frames=1, debug=False)

    def reset(self, rng: jax.Array) -> State:
        q = self.sys.key_qpos
        v = jnp.zeros(self.sys.qd_size())
        act = jnp.zeros(self.action_size)

        state = self.pipeline_init(q, v, act)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        next_state = self.pipeline_step(state, action)
        return next_state


if __name__ == '__main__':
    # =========INFO=========
    # model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
    # data = mujoco.MjData(model)

    # print("Number of joints:", model.njnt)
    # print("Number of actuators:", model.nu)
    # print("Number of bodies:", model.nbody)
    # print("Number of velocities:", model.nv)

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

    # # print('Joint limits:', joint_lower_limits, "\n", joint_upper_limits)
    # print('Limited joints:', model.jnt_limited)
    # print('Gravity:', model.opt.gravity)
    # print('Actuator ranges:', model.actuator_ctrlrange)
    # print('Key q_pos:', model.key_qpos)

    # print(type(model))

    # =============MINIMAL RENDER===================
    env = Go2Env()
    s = env.reset(jnp.zeros(2))
    for i in range(200):
        s = env.step(s, jnp.zeros(env.sys.act_size()))
        env.render()