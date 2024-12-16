import torch
from utils.others.loss_utils import gmof


class ReprojLoss():
    def __init__(self, camera, sigma):
        self.camera = camera
        self.sigma = sigma
        
    
    def __call__(self, joints, gt_keypoints_2d, gt_keypoints_conf):
        projected_joints = self.camera(joints)
        reproj_loss = (gt_keypoints_conf ** 2) * gmof(projected_joints - gt_keypoints_2d, self.sigma).sum(-1)
        return reproj_loss.sum()
        pass