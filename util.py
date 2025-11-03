import numpy as np

A1_q0 = np.array([0, 0, 0.3] + [1, 0, 0, 0] + [0, np.pi/4, -np.pi/2]*4)
A1_joint_lb = np.array([-0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653, -0.802851, -1.0472, -2.69653])
A1_joint_ub = np.array([0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298, 0.802851, 4.18879, -0.916298])

control_freq = 200 # Hz
dt_control = 0.005 # s
dt_sim = 0.001

gravity = np.array([0, 0, -9.806])

default_cam_config = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}

def R_from_quat(quat) -> np.ndarray:
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def projected_gravity(quat) -> np.ndarray:
    '''Return the normalized gravity vector projected into the body frame using quaternion.'''
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    gx = 2*(x*z - w*y) * gravity[2]
    gy = 2*(y*z + w*x) * gravity[2]
    gz = (w*w - x*x - y*y + z*z) * gravity[2]

    vec = np.array([gx, gy, gz])
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

weights = {
    'base_v_xy': 1.0,
    'sigma_v_xy': 0.25,
    'base_v_z': -2.0,
    'angular_xy': -0.05,
    'yaw_rate': 0.5,
    'sigma_yaw': 0.25,
    'projected_gravity': -1.0,
    'effort': -2e-4,
    'joint_accel': -2.5e-7,
    'action_rate': -0.01,
    'contact': -1.0,
    'feet_air_time': 2.0,
    'hip_q': -1.0,
    'thigh_q': -1.0
}

# def euler_from_quat(quat) -> np.ndarray:
#     w, x, y, z = quat[0], quat[1], quat[2], quat[3]
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll_x = np.arctan2(t0, t1)

#     t2 = +2.0 * (w * y - z * x)
#     t2 = +1.0 if t2 > +1.0 else t2
#     t2 = -1.0 if t2 < -1.0 else t2
#     pitch_y = np.arcsin(t2)

#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw_z = np.arctan2(t3, t4)

#     return np.array([roll_x, pitch_y, yaw_z])

# def projected_gravity_2(quat) -> np.ndarray:
#     euler_orientation = euler_from_quat(quat)
#     projected_gravity_not_normalized = (np.dot(gravity, euler_orientation) * euler_orientation)
#     if np.linalg.norm(projected_gravity_not_normalized) == 0:
#         return projected_gravity_not_normalized
#     return projected_gravity_not_normalized / np.linalg.norm(projected_gravity_not_normalized)



