import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco
from robot_descriptions import go2_mj_description
from mujoco_playground._src.mjx_env import MjxEnv, State, make_data, step
from ml_collections import config_dict

class Go2Env(MjxEnv):
    def __init__(self):
        self._xml_path = go2_mj_description.MJCF_PATH
        self._mj_model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
        self._mjx_model = mjx.put_model()
        cfg = config_dict.ConfigDict()
        super().__init__(cfg)

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
    
    def make_state(self) -> State:
        state = State()
        state.obs
        state.reward
        state.done
        state.metrics
        state.info
        state.data = make_data(
            self.mj_model
        )
    
    def reset(self, rng: jax.Array) -> State:
        pass
    
    def step(self, state: State, action: jax.Array) -> State:
        pass
    
    


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