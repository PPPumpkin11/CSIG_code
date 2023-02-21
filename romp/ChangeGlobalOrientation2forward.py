import torch
import numpy as np

file_list = ['./a01_c01', './a01_c02']
Pi = 3.1415926
Pi_axis = torch.from_numpy(np.array([Pi, 0, 0]))

def euler_angles_to_axis_angle(rot_vecs):

    rot_angle = torch.norm(rot_vecs + 1e-8, dim=0, keepdim=True)
    rot_axis = rot_vecs / rot_angle

    return rot_axis, rot_angle

for name in file_list:

    file_name = name + '.npy'
    poses = np.load(file_name, allow_pickle=True)
    poses = poses.astype(np.float32)
    SmplParams_name = name + '_forward.npy'
    print(name)

    for frame in range(0,poses.shape[0]):
        rot_vec = poses[frame, :3].astype(np.float32)
        rot_vec = torch.from_numpy(rot_vec)

        rot_axis, rot_angle = euler_angles_to_axis_angle(rot_vec)
        rot_vec_forward = Pi_axis*rot_angle
        poses[frame, :3] = rot_vec_forward
    np.save(SmplParams_name,poses)