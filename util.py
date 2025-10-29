import numpy as np

### A1 state
# position: [quat, translation, joints]
# velovity: [rotational, translational, joints]
A1_q0 = np.array([1, 0, 0, 0] + [0, 0, 0.3] + [0, np.pi/4, -np.pi/2] * 4)

A1_torque_limit = 33.5

A1_joint_lb = np.array([-0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653])
A1_joint_ub = np.array([0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298])

time_step = 0.005
