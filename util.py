import numpy as np

### A1 state
# position: [quat, translation, joints]
# velovity: [rotational, translational, joints] 0.3
A1_q0 = np.array([1, 0, 0, 0] + [0, 0, 0.35] + [0, np.pi/4, -np.pi/2] * 4)
state_names = ['a1_base_qw', 'a1_base_qx', 'a1_base_qy', 'a1_base_qz', 'a1_base_x', 'a1_base_y', 'a1_base_z', 'a1_FR_hip_joint_q', 'a1_FR_thigh_joint_q', 'a1_FR_calf_joint_q', 'a1_FL_hip_joint_q', 'a1_FL_thigh_joint_q', 'a1_FL_calf_joint_q', 'a1_RR_hip_joint_q', 'a1_RR_thigh_joint_q', 'a1_RR_calf_joint_q', 'a1_RL_hip_joint_q', 'a1_RL_thigh_joint_q', 'a1_RL_calf_joint_q', 'a1_base_wx', 'a1_base_wy', 'a1_base_wz', 'a1_base_vx', 'a1_base_vy', 'a1_base_vz', 'a1_FR_hip_joint_w', 'a1_FR_thigh_joint_w', 'a1_FR_calf_joint_w', 'a1_FL_hip_joint_w', 'a1_FL_thigh_joint_w', 'a1_FL_calf_joint_w', 'a1_RR_hip_joint_w', 'a1_RR_thigh_joint_w', 'a1_RR_calf_joint_w', 'a1_RL_hip_joint_w', 'a1_RL_thigh_joint_w', 'a1_RL_calf_joint_w']

standing_torque = np.array([-0.02848178, 0.43561059, 4.29359326, -0.02848178, 0.38919193, 4.41591229, -0.04882372, 0.23277197, 4.6992705, -0.04882372, 0.19840704, 4.79748208])

time_step = 0.01
time_limit = 60**5
speed = 1.0
torque_scale = 10
