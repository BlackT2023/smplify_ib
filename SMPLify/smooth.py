import torch
from utils.others.loss_utils import axis2angle


class SmoothLoss():
    def __init__(self,
                 pose_smooth_weight=100,
                 shape_smooth_weight=1e6,
                 transl_smooth_weight=1e5,
                 joints_vel_weight=1e3,
                 verts_vel_weight=10,
                 joints_accel_weight=3e3,
                 verts_accel_weight=10,
                 dtype=torch.float32,
                 device='cuda'):
        self.device = device
        self.dtype = dtype
        self.xyz_weight = torch.tensor([3, 3, 10]).to(device).to(dtype)
        
        self.pose_smooth_weight = pose_smooth_weight
        self.shape_smooth_weight = shape_smooth_weight
        self.transl_smooth_weight = transl_smooth_weight
        self.joints_vel_weight = joints_vel_weight
        self.joints_accel_weight = joints_accel_weight
        self.verts_vel_weight = verts_vel_weight
        self.verts_accel_weight = verts_accel_weight
    
    
    def __call__(self,
                 betas,
                 global_orient,
                 body_pose,
                 joints,
                 transl,
                 verts,
                 vel_mask):
        # gt vel
        gt_vel_mask = ~vel_mask
        
        kinematics_smooth_loss = self.kinematics_smooth(joints=joints,
                                                        verts=verts,
                                                        transl=transl,
                                                        gt_vel_mask=gt_vel_mask)
        angle_smooth_loss = self.angle_smooth(global_orient=global_orient,
                                              body_pose=body_pose,
                                              gt_vel_mask=gt_vel_mask)
        shape_smooth_loss = self.shape_smooth(betas)
        return kinematics_smooth_loss + angle_smooth_loss + shape_smooth_loss
    
    
    def kinematics_smooth(self, joints, verts, transl, gt_vel_mask):
        batch_size = joints.shape[0]
        still_mask = torch.ones([batch_size, joints.shape[1]],
                                dtype=self.dtype).to(self.device).requires_grad_(False) * 1
        still_mask[:, :-1][gt_vel_mask] = 10
        still_mask[:, -1] = 10
        # joints velocity loss
        joints_velocity = (
                ((joints[1:] - joints[:-1]) * self.xyz_weight)
                ** 2
        ).sum(-1)
        joints_velocity_loss = (joints_velocity * still_mask[:-1]).sum()
        # joints acceleration loss
        joints_accel = (
                ((2 * joints[1:-1] - joints[:-2] - joints[2:]) * self.xyz_weight)
                ** 2
        ).sum(-1)
        joints_accel_loss = (joints_accel * still_mask[1:-1]).sum()
        # vertices velocity loss
        verts_velocity = torch.norm((verts[1:] - verts[:-1]) * self.xyz_weight) ** 2
        # vertices acceleration loss
        verts_accel = torch.norm(((2 * verts[1:-1] - verts[:-2] - verts[2:]) * self.xyz_weight)) ** 2
        # translation smooth loss
        transl_smooth_loss = (((transl[1:] - transl[:-1]) * self.xyz_weight) ** 2).sum()
        
        return joints_velocity_loss * self.joints_vel_weight + \
               joints_accel_loss * self.joints_accel_weight + \
               verts_velocity * self.verts_vel_weight + \
               verts_accel * self.verts_accel_weight + \
               transl_smooth_loss * self.transl_smooth_weight
    
    
    def angle_smooth(self, global_orient, body_pose, gt_vel_mask):
        batch_size = global_orient.shape[0]
        global_angle = axis2angle(global_orient.view(-1, 3)).view(batch_size, 3)
        body_angle = axis2angle(body_pose.view(-1, 3)).view(batch_size, -1)

        pose_weight = torch.ones_like(body_angle).to(self.device).requires_grad_(False) * 15
        pose_weight[:, 6:18] *= 3
        pose_weight[:, 24:27] *= 3
        # heavy weight on ankle
        pose_weight[:, 18:24] *= 5

        w = 5
        # elbow = [45, 46, 47, 48, 49, 50]
        pose_weight[~gt_vel_mask[:, 5], 45:48] = w
        pose_weight[~gt_vel_mask[:, 6], 48:51] = w
        # hand = [51, 52, 53, 54, 55, 56]
        #pose_weight[~gt_vel_mask[:, 7], 51:54] = w
        pose_weight[~gt_vel_mask[:, 7], 57:60] = w
        pose_weight[~gt_vel_mask[:, 7], 63:66] = w

        #pose_weight[~gt_vel_mask[:, 8], 54:57] = w
        pose_weight[~gt_vel_mask[:, 8], 60:63] = w
        pose_weight[~gt_vel_mask[:, 8], 66:69] = w
        # knee = [0, 1, 2, 3, 4, 5]
        pose_weight[~gt_vel_mask[:, 11], 0:3] = w
        pose_weight[~gt_vel_mask[:, 12], 3:6] = w
        # feet = [6, 7, 8, 9, 10, 11]
        pose_weight[~gt_vel_mask[:, 13], 9:12] = w * 3
        pose_weight[~gt_vel_mask[:, 14], 12:15] = w * 3
        # torso, head more smooth
        # torso = [6, 7, 8, 15, ]
        # pose_weight[:, 6:9] = 50 # low torso
        # pose_weight[:, 15:18] = 50 # mid torso
        # pose_weight[:, 18:24] = 50 # feet
        # pose_weight[:, 24:27] = 50 # upper torso
        # pose_weight[:, 33:36] = 50 # head
        # pose_weight[:, 42:45] = 50 # head
        # pose_weight[:, 36:42] = 50 # shoulder
        # pose_weight[:, 51:] = 1  # hand

        smooth_pose_loss = torch.norm((body_angle[1:] - body_angle[:-1]) * pose_weight[:-1]) ** 2 + \
                    (torch.norm(global_angle[1:] - global_angle[:-1]) ** 2).sum() * 0.1
        return smooth_pose_loss * self.pose_smooth_weight
    
    
    def shape_smooth(self, betas):
        shape_constant_loss = torch.norm(betas[1:] - betas[:-1]) ** 2
        return shape_constant_loss * self.shape_smooth_weight
