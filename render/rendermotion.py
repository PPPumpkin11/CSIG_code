import pickle as pkl
import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
import torch
from renderer import get_renderer

#---------------------
# smpl model
from smpl_wrapper import SMPLWrapper

def euler_angles_to_axis_angle(rot_vecs):
    rot_angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_axis = rot_vecs / rot_angle

    return rot_axis, rot_angle

def get_rotation(theta=np.pi/3):
    import src.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()


def render_video(meshes, key, action, renderer, savepath, background, cam=(0.75, 0.75, 0, 0.10), color=[0.11, 0.53, 0.8]):
    writer = imageio.get_writer(savepath, fps=30)
    # center the first frame
    meshes = meshes - meshes[0].mean(axis=0)
    # matrix = get_rotation(theta=np.pi/4)
    # meshes = meshes[45:]
    # meshes = np.einsum("ij,lki->lkj", matrix, meshes)
    imgs = []
    for mesh in tqdm(meshes, desc=f"Visualize {key}, action {action}"):
        img = renderer.render(background, mesh, cam, color=color)
        imgs.append(img)
        # show(img)

    imgs = np.array(imgs)
    masks = ~(imgs/255. > 0.96).all(-1)

    coords = np.argwhere(masks.sum(axis=0))
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    for cimg in imgs[:, y1:y2, x1:x2]:
        writer.append_data(cimg)
    writer.close()

def render_motion(video_name):

    savefolder = './ActionVideo'
    posefolder = './ActionPoseParameters/'
    key = '0'
    action = '1'

    print(video_name)
    poses = np.load(posefolder + video_name + '.npy', allow_pickle=True)
    videos = poses

    width = 1024
    height = 1024
    background = np.zeros((height,width,3))
    renderer = get_renderer(width,height)
    smpl_wrapper = SMPLWrapper()

    ActionPosePerFrame = {}

    path = os.path.join(savefolder, "{}.mp4".format(video_name))
    Action_Pose = torch.from_numpy(poses)

    for frameInd in range(Action_Pose.shape[0]):

        ActionPosePerFrame['poses'] = Action_Pose[frameInd, :].view(1,-1)
        Smpl_vertex_3dkpt = smpl_wrapper.forward(ActionPosePerFrame)
        v_template = Smpl_vertex_3dkpt['verts'].squeeze(0).cpu().detach().numpy()
        VertsInOneFrame = v_template[np.newaxis, :]
        if frameInd >0:
            ActionPosePerFrameAll = np.concatenate((ActionPosePerFrameAll, VertsInOneFrame),axis=0)
        else:
            ActionPosePerFrameAll = VertsInOneFrame

    render_video(ActionPosePerFrameAll, key, action, renderer, path, background)


def main():
#--------------------
# 读取姿态数据，并可视化动作序列
#--------------------
    file_list = ['a01_p_c01_forward']

    for video in file_list:
        render_motion(video)



if __name__ == "__main__":
    main()
