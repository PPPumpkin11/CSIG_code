import torch
import torch.nn as nn
import numpy as np 

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# import config
# from config import args
import constants
from smpl import SMPL
from projection import vertices_kp3d_projection
from rot_6D import rot6D_to_angular

class SMPLWrapper(nn.Module):
    def __init__(self):
        super(SMPLWrapper,self).__init__()
        self.smpl_model = SMPL('./models/smpl/SMPL_NEUTRAL.pkl', J_reg_extra9_path='./models/smpl/J_regressor_extra.npy', J_reg_h36m17_path='./models/smpl/J_regressor_h36m.npy', \
            batch_size= 1, model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False,\
            ).cuda() #dtype=torch.float16 if args().model_precision=='fp16' else torch.float32
        self.part_name = ['cam', 'global_orient', 'body_pose', 'betas']
        self.part_idx = [3, 6,  (22-1)*6,       10]

        self.unused_part_name = ['left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
        self.unused_part_idx = [        15,                  15,           3,          3,            3,          10]
        
        self.kps_num = 25
        self.params_num = np.array(self.part_idx).sum()
        self.global_orient_nocam = torch.from_numpy(constants.global_orient_nocam).unsqueeze(0)
        self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()
        self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()

    def forward(self, SmplParams):

        smpl_outs = self.smpl_model(**SmplParams, return_verts=True, return_full_pose=True)
        outputs = {}
        outputs.update({'params': SmplParams, **smpl_outs})
        
        return outputs

    def recalc_outputs(self, params_dict, meta_data):
        smpl_outs = self.smpl_model.single_forward(**params_dict, return_verts=True, return_full_pose=True)
        outputs = {'params': params_dict, **smpl_outs}
        outputs.update(vertices_kp3d_projection(outputs,meta_data=meta_data,presp=args().perspective_proj))
        outputs = set_items_float(outputs)
        
        return outputs

def set_items_float(out_dict):
    items = list(out_dict.keys())
    for item in items:
        if isinstance(out_dict[item], dict):
            out_dict[item] = set_items_float(out_dict[item])
        elif isinstance(out_dict[item], torch.Tensor):
            out_dict[item] = out_dict[item].float()
    return out_dict

def euler_angles_to_axis_angle(rot_vecs):
    rot_angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_axis = rot_vecs / rot_angle

    return rot_axis, rot_angle
