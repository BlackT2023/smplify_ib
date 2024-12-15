import torch
from utils.others.loss_utils import axis2angle


class ConsistencyLoss():
    def __init__(self,
                 angle_consistency_weight=40,
                 joints_consistency_weight=1e5,
                 transl_consistency_weight=1e10,
                 betas_consistency_weight=1e7,
                 verts_consistency_weight=1e4,
                 device='cuda',
                 dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.xyz_weight = torch.tensor([3, 3, 5]).to(self.device)
        
        self.angle_consistency_weight = angle_consistency_weight
        self.joints_consistency_weight = joints_consistency_weight
        self.transl_consistency_weight = transl_consistency_weight
        self.betas_consistency_weight = betas_consistency_weight
        self.verts_consistency_weight = verts_consistency_weight
        pass
    
    
    def __call__(self,
                 prev_result,
                 body_pose,
                 global_orient,
                 model_joints,
                 betas,
                 transl,
                 verts,
                 segments,
                 idx):
        batch_size = betas.shape[0]
        
        prev_window_l = batch_size - prev_result['idx'][1] + idx[0]
        present_window_r = prev_result['idx'][1] - idx[0]

        # angle-diff
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        full_angle = axis2angle(full_pose.view(-1, 3)).view(batch_size, -1)
        full_angle_prev = axis2angle(prev_result['full_pose'].view(-1, 3)).view(batch_size, -1)
        full_angle_diff = full_angle[:present_window_r] - full_angle_prev[prev_window_l:]
        torso_angle_diff = torch.cat([full_angle_diff[:, :12],
                                    full_angle_diff[:, 18:39],
                                    full_angle_diff[:, 45:48]], dim=1)
        angle_diff = torch.norm(torso_angle_diff) ** 2

        # points-diff
        torso_verts, _, _ = segments.get_unclosed_segments(vertices=verts,
                                                        smooth_joints=None,
                                                        names=['head', 'torso_upperarm', 'torso_thigh'],
                                                        requires_center=False)
        prev_torso_verts, _, _ = segments.get_unclosed_segments(vertices=prev_result['vertices'],
                                                                smooth_joints=None,
                                                                names=['head', 'torso_upperarm', 'torso_thigh'],
                                                                requires_center=False)
        torso_joints = model_joints[:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15]]
        prev_torso_joints = prev_result['model_joints'][:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15]]
        joints_diff = torch.norm(
            (torso_joints[:present_window_r] - prev_torso_joints[prev_window_l:]) * self.xyz_weight) ** 2
        vertices_diff = torch.norm(
            (torso_verts[:present_window_r] - prev_torso_verts[prev_window_l:]) * self.xyz_weight) ** 2
        transl_diff = torch.norm((transl[:present_window_r] - prev_result['transl'][prev_window_l:]) * self.xyz_weight) ** 2

        # betas-diff
        betas_diff = torch.norm(betas[:present_window_r] - prev_result['betas'][prev_window_l:]) ** 2

        consistency_loss = angle_diff * self.angle_consistency_weight + \
                           joints_diff * self.joints_consistency_weight + \
                           vertices_diff * self.verts_consistency_weight + \
                           transl_diff * self.transl_consistency_weight + \
                           betas_diff * self.betas_consistency_weight