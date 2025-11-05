import numpy as np

dt_control = 0.01 # 100Hz
dt_sim = 0.002

default_cam_config = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
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



