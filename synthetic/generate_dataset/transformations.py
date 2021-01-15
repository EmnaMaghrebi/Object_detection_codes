import numpy as np
from cameraCharacteristics import CameraPose, CameraDistance

def spatial_transform_Matrix(t_x=0.0, t_y=0.0, t_z=0.0, scale_x=1.0, scale_y=1.0, scale_z=1.0, roll=0.0, pitch=0.0, yaw=0.0, zoom=1.0, scale=1.0):
    translation_vector = np.array([t_x/zoom, t_y/zoom,t_z/zoom, 1])
    translation = np.eye(4)
    translation[:,3] = translation_vector
    scaling_vector = np.array([scale_x*scale, scale_y*scale, scale_z*scale, 1])
    scaling = np.eye(4) * scaling_vector
    R_roll = np.array([
        [1.0,           0.0,           0.0, 0.0],
        [0.0, np.cos(roll),-np.sin(roll), 0.0],
        [0.0, np.sin(roll), np.cos(roll), 0.0],
        [0.0,           0.0,           0.0, 1.0]
    ])
    R_pitch = np.array([
        [ np.cos(pitch), 0.0, np.sin(pitch), 0.0],
        [            0.0, 1.0,            0.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch), 0.0],
        [            0.0, 0.0,            0.0, 1.0]
    ])
    R_yaw = np.array([
        [np.cos(yaw),-np.sin(yaw), 0.0, 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0, 0.0],
        [         0.0,          0.0, 1.0, 0.0],
        [         0.0,          0.0, 0.0, 1.0]
    ])
    return translation @ scaling @ R_yaw @ R_pitch @ R_roll

def lookAt(view=CameraPose.NONE, distance=CameraDistance.MEDIUM, alpha=0.0, beta=0.0, force_t_z=-500.0, force=False, at_x=0.0, at_y=0.0, at_z=0.0):
    assert - np.pi * 2.0 < alpha < np.pi * 2.0
    assert 0 <= beta < np.pi
    if distance in [item for item in CameraDistance]:
        distance = distance.value / 10
    beta = 0.0 if view == CameraPose.TOP else (np.pi/2.0 if view == CameraPose.SIDE else beta)
    t_z = distance * np.cos(beta) if 0 <= beta <= np.pi/2 else -1*distance * np.sin(beta - np.pi/2)
    t_x = distance * np.sin(alpha) * np.sin(beta) if 0 <= beta <= np.pi/2 else distance * np.sin(alpha) * np.cos(beta - np.pi/2)
    t_y = -1 * distance * np.cos(alpha) * np.sin(beta) if 0 <= beta <= np.pi/2 else -1 * distance * np.cos(alpha) * np.cos(beta - np.pi/2)
    t_z = force_t_z if (force_t_z > t_z or force) else (0.0 if -1e-8 < t_z < 1e-8 else t_z)
    t_y = 0.0 if -1e-8 < t_y < 1e-8 else t_y
    t_x = 0.0 if -1e-8 < t_x < 1e-8 else t_x
    M = spatial_transform_Matrix(t_x=t_x + at_x, t_y=t_y + at_y, t_z=max(0.0, t_z + at_z), yaw=alpha, roll=beta)
    return M, [t_x + at_x, t_y + at_y, max(0.0, t_z + at_z), distance, alpha, beta]