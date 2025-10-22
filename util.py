import numpy as np

### A1 state
# position: [quat, translation, joints]
# velovity: [rotational, translational, joints]
A1_q0 = [1, 0, 0, 0] + [0, 0, 0.3] + [0, np.pi/4, -np.pi/2] * 4

standing_torque = [-0.02848178, 0.43561059, 4.29359326, -0.02848178, 0.38919193, 4.41591229, -0.04882372, 0.23277197, 4.6992705, -0.04882372, 0.19840704, 4.79748208]

time_step = 0.005

torque_scaler = 10
