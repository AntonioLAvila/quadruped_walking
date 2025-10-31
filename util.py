import numpy as np

### A1 state
# position: [quat, translation, joints]
# velovity: [rotational, translational, joints]
A1_q0 = np.array([0, 0, 0.3] + [1, 0, 0, 0] + [0, np.pi/4, -np.pi/2]*4)

A1_torque_limit = 33.5

A1_joint_lb = np.array([-0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653])
A1_joint_ub = np.array([0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298])

control_freq = 200 # Hz
dt_control = 0.005
dt_sim = 0.001


default_cam_config = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}