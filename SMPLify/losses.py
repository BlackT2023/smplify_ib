import setup
import torch
import torch.nn as nn
import pickle
import numpy as np

from .bed_contact import BedContactLoss
from .consistency import ConsistencyLoss
from .gravity import GravityLoss
from .prior import PriorLoss
from .self_contact import SelfContactLoss
from .smooth import SmoothLoss
from .reproj import ReprojLoss
from config import config

class GravityFittingLoss():
    def __init__(self,
                 weights,
                 camera,
                 dist_dict=None,
                 bc_type='simple',
                 sc_type='ours',
                 bed_depth=1.66,
                 # sc
                 segments=None,
                 modified_mask=None,
                 geomask=None,
                 faces=None,
                 search_tree=None,
                 pen_distance=None,
                 device=torch.device('cuda'),
                 dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        self.dist_dict = dist_dict
        self.bc_type = bc_type
        self.sc_type = sc_type
        
        self.reproj_module = ReprojLoss(camera=camera,
                                        sigma=weights['sigma'])
        
        self.consistency_module = ConsistencyLoss(angle_consistency_weight=weights['consistency_angle'],
                                                  joints_consistency_weight=weights['consistency_joints'],
                                                  transl_consistency_weight=weights['consistency_transl'],
                                                  betas_consistency_weight=weights['consistency_betas'],
                                                  verts_consistency_weight=weights['consistency_verts'],
                                                  device=device,
                                                  dtype=dtype)
        self.gravity_module = GravityLoss(dist_dict=dist_dict,
                                          bed_depth=bed_depth,
                                          gravity_beta1=weights['gravity_beta1'],
                                          gravity_beta2=weights['gravity_beta2'],
                                          gravity_inside_torso_alpha=weights['gravity_inside_torso_alpha'],
                                          gravity_inside_torso_beta=weights['gravity_inside_torso_beta'],
                                          device=device,
                                          dtype=dtype)
        self.prior_module = PriorLoss(prior_folder=config.PRIOR_PATH,
                                      dist_dict=dist_dict,
                                      num_gaussians=weights['num_gaussians'],
                                      alpha_head_angle=weights['prior_head_angle'],
                                      alpha_elbow_angle=weights['prior_elbow_angle'],
                                      alpha_knee_angle=weights['prior_knee_angle'],
                                      alpha_limb_angle=weights['prior_limb_angle'],
                                      alpha_torso_angle=weights['prior_torso_angle'],
                                      alpha_forw_torso=weights['prior_torso_forw'],
                                      alpha_back_torso=weights['prior_torso_back'],
                                      beta_torso=weights['prior_torso_beta'],
                                      alpha_pose=weights['prior_pose_alpha'],
                                      alpha_angle=weights['prior_angle_alpha'],
                                      alpha_shape=weights['prior_shape_alpha'],
                                      alpha_torso=weights['prior_torso_alpha'],
                                      device=device,
                                      dtype=dtype)
        self.smooth_module = SmoothLoss(pose_smooth_weight=weights['smooth_pose'],
                                        shape_smooth_weight=weights['smooth_shape'],
                                        transl_smooth_weight=weights['smooth_transl'],
                                        joints_vel_weight=weights['smooth_joints_vel'],
                                        verts_vel_weight=weights['smooth_verts_vel'],
                                        joints_accel_weight=weights['smooth_joints_accel'],
                                        verts_accel_weight=weights['smooth_verts_accel'],
                                        device=device,
                                        dtype=dtype)
        self.bc_module = BedContactLoss(bed_depth=bed_depth,
                                        segments=segments,
                                        bc_euclthres=weights['bc_euclthres'],
                                        bc_inside_alpha=weights['bc_inside_alpha'],
                                        bc_inside_beta=weights['bc_inside_beta'],
                                        bc_outside_alpha=weights['bc_outside_alpha'],
                                        bc_outside_beta=weights['bc_outside_beta'])
        self.sc_module = SelfContactLoss(segments=segments,
                                         modified_mask=modified_mask,
                                         geomask=geomask,
                                         faces=faces,
                                         search_tree=search_tree,
                                         pen_distance=pen_distance,
                                         sc_euclthres=weights['sc_euclthres'],
                                         sc_inside_alpha=weights['sc_inside_alpha'],
                                         sc_inside_beta=weights['sc_inside_beta'],
                                         sc_outside_alpha=weights['sc_outside_alpha'],
                                         sc_outside_beta=weights['sc_outside_beta'],
                                         sc_joint_alpha=weights['sc_joints_alpha'],
                                         sc_joint_beta=weights['sc_joint_beta'],
                                         device=device,
                                         dtype=dtype)
    
    
    def __call__(self,
                 global_orient,
                 body_pose,
                 betas,
                 transl,
                 joints,
                 vertices,
                 vel_mask,
                 gt_keypoints_2d,
                 gt_keypoints_conf,
                 prev_result,
                 idx):
        reproj_loss = self.reproj_module(joints=joints,
                                         gt_keypoints_2d=gt_keypoints_2d,
                                         gt_keypoints_conf=gt_keypoints_conf)
        smooth_loss = self.smooth_module(betas=betas,
                                         global_orient=global_orient,
                                         body_pose=body_pose,
                                         joints=joints,
                                         transl=transl,
                                         verts=vertices,
                                         vel_mask=vel_mask)
        bc_loss = self.bc_module(vertices)
        consistency_loss = self.consistency_module(prev_result=prev_result,
                                                   body_pose=body_pose,
                                                   global_orient=global_orient,
                                                   model_joints=joints,
                                                   betas=betas,
                                                   transl=transl,
                                                   verts=vertices,
                                                   segments=self.segments,
                                                   idx=idx)
        
        pass
    
    

class SelfcontactFittingLoss():
    def __init__(self,
                 weights):
        
        pass
    
    
    def __call__(self,):
        pass
    
    
class CameraFittingLoss():
    def __init__(self,
                 weights,
                 camera,
                 segments,
                 bed_depth=1.66,
                 device=torch.device('cuda'),
                 dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.segments = segments
        
        self.reproj_module = ReprojLoss(camera=camera,
                                        sigma=weights['sigma'])
        
        self.consistency_module = ConsistencyLoss(angle_consistency_weight=weights['consistency_angle'],
                                                  joints_consistency_weight=weights['consistency_joints'],
                                                  transl_consistency_weight=weights['consistency_transl'],
                                                  betas_consistency_weight=weights['consistency_betas'],
                                                  verts_consistency_weight=weights['consistency_verts'],
                                                  device=device,
                                                  dtype=dtype)
        self.smooth_module = SmoothLoss(pose_smooth_weight=weights['smooth_pose'],
                                        shape_smooth_weight=weights['smooth_shape'],
                                        transl_smooth_weight=weights['smooth_transl'],
                                        joints_vel_weight=weights['smooth_joints_vel'],
                                        verts_vel_weight=weights['smooth_verts_vel'],
                                        joints_accel_weight=weights['smooth_joints_accel'],
                                        verts_accel_weight=weights['smooth_verts_accel'],
                                        device=device,
                                        dtype=dtype)
        self.bc_module = BedContactLoss(bed_depth=bed_depth,
                                        segments=None,
                                        bc_euclthres=weights['bc_euclthres'],
                                        bc_inside_alpha=weights['bc_inside_alpha'],
                                        bc_inside_beta=weights['bc_inside_beta'],
                                        bc_outside_alpha=weights['bc_outside_alpha'],
                                        bc_outside_beta=weights['bc_outside_beta'])
    
    
    def __call__(self,
                 global_orient,
                 body_pose,
                 betas,
                 transl,
                 joints,
                 vertices,
                 vel_mask,
                 gt_keypoints_2d,
                 gt_keypoints_conf,
                 prev_result,
                 idx):
        reproj_loss = self.reproj_module(joints=joints,
                                         gt_keypoints_2d=gt_keypoints_2d,
                                         gt_keypoints_conf=gt_keypoints_conf)
        smooth_loss = self.smooth_module(betas=betas,
                                         global_orient=global_orient,
                                         body_pose=body_pose,
                                         joints=joints,
                                         transl=transl,
                                         verts=vertices,
                                         vel_mask=vel_mask)
        bc_loss = self.bc_module(vertices)
        consistency_loss = self.consistency_module(prev_result=prev_result,
                                                   body_pose=body_pose,
                                                   global_orient=global_orient,
                                                   model_joints=joints,
                                                   betas=betas,
                                                   transl=transl,
                                                   verts=vertices,
                                                   segments=self.segments,
                                                   idx=idx)
        return reproj_loss + smooth_loss + bc_loss + consistency_loss