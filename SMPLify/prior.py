import os
import torch

from utils.others.gmm import MaxMixturePrior


class PriorLoss():
    def __init__(self,
                 prior_folder,
                 dist_dict,
                 num_gaussians=8,
                 # angle
                 alpha_head_angle=600,
                 alpha_elbow_angle=600,
                 alpha_knee_angle=450,
                 alpha_limb_angle=1500,
                 alpha_torso_angle=12000,
                 # torso
                 alpha_forw_torso=10,
                 alpha_back_torso=1,
                 beta_torso=0.02,
                 # sum
                 alpha_pose=500,
                 alpha_angle=20,
                 alpha_shape=2.5e5,
                 alpha_torso=1e3,
                 device='cuda',
                 dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.dist_dict = dist_dict
        
        # for angle prior
        self.alpha_head_angle = alpha_head_angle
        self.alpha_elbow_angle = alpha_elbow_angle
        self.alpha_knee_angle = alpha_knee_angle
        self.alpha_limb_angle = alpha_limb_angle
        self.alpha_torso_angle = alpha_torso_angle
        
        # for pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=prior_folder,
                                          num_gaussians=num_gaussians,
                                          dtype=dtype).to(device)
        
        # for torso prior
        self.alpha_forw_torso = alpha_forw_torso
        self.alpha_back_torso = alpha_back_torso
        self.beta_torso = beta_torso
        
        # for sum up
        self.alpha_angle = alpha_angle
        self.alpha_pose = alpha_pose
        self.alpha_shape = alpha_shape
        self.alpha_torso = alpha_torso
    
    
    def __call__(self,
                 betas,
                 body_pose,
                 joints,
                 torso_joints,
                 gt_joints_2d):
        angle_prior_loss = self.angle_prior(body_pose=body_pose, gt_joints_2d=gt_joints_2d)
        torso_prior_loss = self.torso_prior(joints=joints, torso_joints=torso_joints)
        pose_prior_loss = self.pose_prior(betas=betas, body_pose=body_pose)
        shape_prior_loss = self.shape_prior(betas=betas)
        return angle_prior_loss * self.alpha_angle + \
               torso_prior_loss * self.alpha_torso + \
               pose_prior_loss * self.alpha_pose + \
               shape_prior_loss * self.alpha_shape
    
    
    def angle_prior(self, body_pose, gt_joints_2d):
        # head, forbid facing-down and severe facing-up
        head_err = body_pose[:, [33, 42]] * 4
        facing_down = head_err > 0
        head_prior = torch.exp(facing_down * (head_err - 0.5)).sum() + torch.exp((~facing_down) * head_err * (-1)).sum() + \
                    torch.exp(torch.abs(body_pose[:, [34, 35, 43, 44]]) * 2 - 1).sum()

        # elbow and shoulder, forbid impossible angle and uncomfortable rotation
        # elbow_shoulder_err = pose[:, [37, 40, 46, 49, 52, 55]] * torch.tensor([1, -1, 1, -1, 1, -1], device=device, dtype=dtype) * 4
        elbow_shoulder_err = body_pose[:, [37, 40]] * torch.tensor([1, -1], device=self.device, dtype=self.dtype) * 4
        elbow_shoulder_mask = elbow_shoulder_err > 0
        rotation_err = torch.abs(body_pose[:, [36, 39, 45, 48]]) - 0.5
        rotation_mask = rotation_err > 0
        elbow_prior = torch.exp(elbow_shoulder_mask * elbow_shoulder_err).sum() + \
                    torch.exp(rotation_mask * rotation_err * 4).sum() + \
                    torch.exp(torch.abs(body_pose[:, [53, 56]]) * 4).sum()

        # knee, forbid impossible angle and severe inside-knee
        knee_err = body_pose[:, [2, 5, 9, 12]] * torch.tensor([-1, 1, -1, -1], device=self.device, dtype=self.dtype) * 4
        knee_mask = knee_err > 0
        rotation_err = torch.abs(body_pose[:, [11, 14]]) * 4
        knee_prior = torch.exp(knee_mask * knee_err).sum() + \
                    torch.exp(rotation_err).sum()
                    
        # hand, feet
        hand_feet_err = body_pose[:, [18, 21, 65, 68]] * torch.tensor([-1, -1, 1, -1], device=self.device, dtype=self.dtype) * 4
        hand_feet_mask = hand_feet_err > 0
        rotation_err = torch.abs(body_pose[:, [19, 20, 22, 23, 57, 58, 60, 61]]) * 4
        # vertical hand
        real_dist = torch.tensor([self.dist_dict['hand'], self.dist_dict['hand']],
                                dtype=self.dtype, device=self.device, requires_grad=False)
        est_dist = torch.norm(gt_joints_2d[:, [5, 6]] - gt_joints_2d[:, [7, 8]],
                            dim=-1)
        cosine = est_dist / real_dist
        vertical_hand = cosine < 0.5
        hand_rotation = torch.abs(body_pose[:, [59, 62]]) * 2 - 1
        hand_mask2 = hand_rotation > 0
        hand_feet_prior = torch.exp(hand_feet_mask * hand_feet_err).sum() + \
                        torch.exp(rotation_err).sum() + \
                        torch.exp((~vertical_hand) * hand_rotation * hand_mask2).sum()
        # torso
        torso_err = torch.abs(body_pose[:, [6, 8, 15, 17, 24, 26]]) * 5
        l_torso = torch.norm(gt_joints_2d[:, 3] - gt_joints_2d[:, 9], dim=-1) / self.dist_dict['ltorso']
        r_torso = torch.norm(gt_joints_2d[:, 4] - gt_joints_2d[:, 10], dim=-1) / self.dist_dict['rtorso']
        torso_mask = (l_torso > 0.9) & (r_torso > 0.9)
        rotation_err = torch.abs(body_pose[:, [7, 16, 25]]) * 10
        torso_prior = torch.exp(torso_mask.unsqueeze(-1) * torso_err).sum() + \
                    torch.exp(rotation_err).sum() * 0
        return head_prior * self.alpha_head_angle + \
            elbow_prior * self.alpha_elbow_angle + \
            knee_prior * self.alpha_knee_angle + \
            hand_feet_prior * self.alpha_limb_angle + \
            torso_prior * self.alpha_torso_angle
    
    
    def torso_prior(self, joints, torso_joints):
        # lshoulder, hips
        x_lhip2rhip = joints[:, 10] - joints[:, 9]
        y_lsoulder2lhip = joints[:, 9] - joints[:, 3]
        n1 = torch.cross(x_lhip2rhip, y_lsoulder2lhip, dim=-1)
        d1 = torso_joints - joints[:, 9].unsqueeze(1)
        proj_1 = (d1 * n1.unsqueeze(1)).sum(-1) / torch.norm(n1, dim=-1).unsqueeze(-1)
        mask1 = (proj_1 > 0)
        mask2 = proj_1 < -0.1
        loss1 = (mask1 * torch.exp(proj_1 / self.beta_torso)).sum() * self.alpha_forw_torso + \
                (mask2 * torch.exp((proj_1 + 0.1) / self.beta_torso * (-1))).sum() * self.alpha_back_torso

        # rshoulder, hips
        x_rhip2lhip = joints[:, 9] - joints[:, 10]
        y_rshoulder2rhip = joints[:, 4] - joints[:, 10]
        n2 = torch.cross(x_rhip2lhip, y_rshoulder2rhip, dim=-1)
        d2 = torso_joints - joints[:, 10].unsqueeze(1)
        proj_2 = (d2 * n2.unsqueeze(1)).sum(-1) / torch.norm(n2, dim=-1).unsqueeze(-1)
        mask1 = (proj_2 > 0)
        mask2 = proj_2 < -0.1
        loss2 = (mask1 * torch.exp(proj_2 / self.beta_torso)).sum() * self.alpha_forw_torso + \
                (mask2 * torch.exp((proj_2 + 0.1) / self.beta_torso * (-1))).sum() * self.alpha_forw_torso

        return loss1 + loss2
    
    
    def pose_prior(self, betas, body_pose):
        return self.pose_prior(body_pose, betas)
    
    
    def shape_prior(self, betas):
        return torch.norm(betas) ** 2
