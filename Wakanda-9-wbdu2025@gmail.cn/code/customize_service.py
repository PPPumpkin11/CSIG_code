
# import log
# from model_service.pytorch_model_service import PTServingBaseService
import torch.nn.functional as F

import torch.nn as nn
import torch
from torch.autograd import Variable
import json

import numpy as np
import random

# logger = log.getLogger(__name__)

from model.example.utils import matrix_to_rotation_6d,quaternion_to_matrix,axis_angle_to_matrix,axis_angle_to_quaternion
from model.example.Net import Net
from models.transform1 import (
    axis_angle_to_matrix,
    # axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)
import torchvision.transforms as transforms
import os

# quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
# 从输入的序列中随机选取固定帧数作为网络输入
def SelectNetworkInput(ActionPose, ActionLength,rand=False):

    if rand:
        # step = np.random.choice([1,1.5,2,2.5,3])
        step = np.random.choice([1.5,2,2.5,3])
    else:
        step = 2  # 固定隔帧抽取
    nframes = ActionPose.shape[0]

    # 从输入序列中随机选择ActionLength帧数据作为输入
    lastone = step * (ActionLength - 1)
    shift_max = nframes - lastone - 1  # ActionLength条件下序列中最后一个有效帧序号
    shift_max = int(shift_max)
    shift = random.randint(0, max(0, shift_max - 1))  # 在0到最后一个有效帧之间生成一个随机偏移量
    # shift = 0#random.randint(0, max(0, shift_max - 1))  # 在0到最后一个有效帧之间生成一个随机偏移量
    frame_idx = shift + np.arange(0, lastone + 1, step)
    # print(frame_idx)
    frame_idx = frame_idx.astype('int')
    # print(ActionPose.shape)
    pose = ActionPose[frame_idx, :].astype(np.float32)
    pose = torch.from_numpy(pose).reshape(-1, 24, 3)  # 这里

    # 进行姿态格式转换: 三元轴角式->四元数->旋转矩阵->rot6D  data.shape=[60,24,6]
    # input_pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))

    input_pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose))
    input_pose = input_pose.permute(1, 2, 0).contiguous()  # 将数据维度变成[24,6,ActionLength]
    input_pose = input_pose.float()
    return input_pose
mask = torch.from_numpy(np.ones([24,],dtype='float32'))#.cuda()
mask[0] = 0
def SelectNetworkInput1(ActionPose, ActionLength,rand=False):

    if rand:
        # step = np.random.choice([1,1.5,2,2.5,3])
        step = np.random.choice([1.5,2,2.5,3])
    else:
        step = 2  # 固定隔帧抽取
    nframes = ActionPose.shape[0]

    # 从输入序列中随机选择ActionLength帧数据作为输入
    lastone = step * (ActionLength - 1)
    shift_max = nframes - lastone - 1  # ActionLength条件下序列中最后一个有效帧序号
    shift_max = int(shift_max)
    shift = random.randint(0, max(0, shift_max - 1))  # 在0到最后一个有效帧之间生成一个随机偏移量
    # shift = 0#random.randint(0, max(0, shift_max - 1))  # 在0到最后一个有效帧之间生成一个随机偏移量
    frame_idx = shift + np.arange(0, lastone + 1, step)
    # print(frame_idx)
    frame_idx = frame_idx.astype('int')
    # print(ActionPose.shape)
    pose = ActionPose[frame_idx, :].astype(np.float32)
    pose = torch.from_numpy(pose).reshape(-1, 24, 3)  # 这里

    pose = pose*mask[None,:,None]
    # 进行姿态格式转换: 三元轴角式->四元数->旋转矩阵->rot6D  data.shape=[60,24,6]
    # input_pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
    # input_pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose))
    # input_pose = axis_angle_to_matrix(pose)
    # input_pose = input_pose.permute(1, 2, 0).contiguous()  # 将数据维度变成[24,6,ActionLength]
    # input_pose = input_pose.float()
    # input_pose_ = input_pose.permute(2, 0, 1).contiguous()
    # input_pose_ = rotation_6d_to_matrix(input_pose)
    # input_pose_ = input_pose
    # npy_random_select = quaternion_to_axis_angle(matrix_to_quaternion(input_pose_))
    input_pose = pose.permute(1, 2, 0).contiguous()

    return input_pose
# class PTVisionService(PTServingBaseService):
#
#     def __init__(self, model_name, model_path):
#         # 调用父类构造方法
#         super(PTVisionService, self).__init__(model_name, model_path)
#         # 调用自定义函数加载模型
#         self.model = ActionRecogModel(model_path)
#
#     def _preprocess(self, data):
#         ActionLength = 60
#         preprocessed_data = {}
#         for k, v in data.items():
#             for file_name, file_content in v.items():
#
#                 print('t\Loading Action Poses: %s' % file_name)
#
#                 ActionPose = np.load(file_content, allow_pickle=True)
#                 input_pose = SelectNetworkInput(ActionPose, ActionLength)
#                 data_input = Variable(input_pose, requires_grad=True)
#
#                 if torch.cuda.is_available():
#                     data_input = data_input.cuda()
#                 preprocessed_data[k] = data_input
#
#         return preprocessed_data
#
#     def _postprocess(self, data):
#         for k, v in data.items():
#             class_scores = {'action_class': v.tolist()}
#
#         #print(f'postprocess class_scores key:{class_scores.keys()}')
#         return class_scores
#
#     def _inference(self, data):
#         #print(f"model input data:{data.}")
#         output = {}
#         for k, v in data.items():
#             output[k] = self.model(v)
#             output[k] = output[k].clone().squeeze(0).detach().cpu().numpy()
#
#         return output


def ActionRecogModel(model_path, **kwargs):
    ActionLength = 60  # 比赛规定输入序列长度为60帧
    # 生成网络
    model = Net(ActionLength)
    # 加载模型
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
    # CPU或者GPU映射
    model.to(device)
    # 声明为推理模式
    model.eval()

    return model

