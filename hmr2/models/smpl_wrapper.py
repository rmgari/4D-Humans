import torch
import numpy as np
import pickle
from typing import Optional
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput


class SMPL(smplx.SMPLHLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, update_hips: bool = False, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(SMPL, self).__init__(*args, **kwargs)
        smplh_to_openpose = [52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 20, 34, 35,36, 63, 22, 23, 24, 64, 25, 26, 27, 65, 31, 32, 33, 66, 28, 29, 30, 67, 21, 49, 50, 51, 68, 37, 38, 39, 69, 40, 41, 42, 70, 46, 47, 48, 71, 43, 44, 45, 72, ]    
            
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('joint_map', torch.tensor(smplh_to_openpose, dtype=torch.long))
        self.update_hips = update_hips

    def forward(self, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        if self.update_hips:
            joints[:,[9,12]] = joints[:,[9,12]] + \
                0.25*(joints[:,[9,12]]-joints[:,[12,9]]) + \
                0.5*(joints[:,[8]] - 0.5*(joints[:,[9,12]] + joints[:,[12,9]]))
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
            body = joints[:, :25]
            hands = joints[:, 25:]
            joints = torch.cat([body, extra_joints, hands], dim=1)

        smpl_output.joints = joints
        return smpl_output
