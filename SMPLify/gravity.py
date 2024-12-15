import torch
from utils.others.loss_utils import polygon_method

class GravityLoss():
    def __init__(self,
                 dist_dict,
                 bed_depth=1.66,
                 alpha_inside_torso=1,
                 beta1=0.05,
                 beta2=0.5,
                 beta_inside_torso=1.5,
                 device='cuda',
                 dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        self.dist_dict = dist_dict
        self.bed_depth = bed_depth
        
        self.alpha_inside_torso = alpha_inside_torso
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta_inside_torso = beta_inside_torso
        
    
    def __call__(self,
                 model_joints,
                 gt_joints_2d,
                 vel_mask):
        batch_size = model_joints.shape[0]
        # dist mask, assume the human is lying on the bed
        real_dist = torch.tensor([self.dist_dict['elbow'], self.dist_dict['elbow'], self.dist_dict['hand'], self.dist_dict['hand'],
                                self.dist_dict['knee'], self.dist_dict['knee'], self.dist_dict['feet'], self.dist_dict['feet']],
                                dtype=self.dtype, device=self.device, requires_grad=False)
        est_dist = torch.norm(gt_joints_2d[:, [3, 4, 5, 6, 9, 10, 11, 12]] - gt_joints_2d[:, [5, 6, 7, 8, 11, 12, 13, 14]],
                            dim=-1)
        cosine = est_dist / real_dist

        hand_gravity_err = -model_joints[:, [7, 8], -1]
        hand_gravity_weight = torch.ones_like(hand_gravity_err, device=self.device, dtype=self.dtype, requires_grad=False) / self.beta2
        hand_gravity_weight[cosine[:, 2:4] > 0.55] = 1 / self.beta1 * 2.7

        elbow_gravity_err = -model_joints[:, [5, 6], -1]
        elbow_gravity_weight = torch.ones_like(elbow_gravity_err, device=self.device, dtype=self.dtype,
                                            requires_grad=False) / self.beta2 / 2
        elbow_gravity_weight[(cosine[:, 0:2] > 0.6) & (cosine[:, 2:4] > 0.6)] = 1 / self.beta1

        feet_gravity_err = -model_joints[:, [13, 14], -1]
        feet_gravity_weight = torch.zeros_like(feet_gravity_err, device=self.device, dtype=self.dtype, requires_grad=False) / self.beta2
        feet_gravity_weight[cosine[:, 6:8] > 0.5] = 1 / self.beta1 * 3

        knee_gravity_err = -model_joints[:, [11, 12], -1]
        knee_gravity_weight = torch.ones_like(knee_gravity_err, device=self.device, dtype=self.dtype, requires_grad=False) / self.beta2 / 2
        knee_gravity_weight[(cosine[:, 6:8] > 0.7) & (cosine[:, 4:6] > 0.7)] = 1 / self.beta1 * 2

        # especially, when hand inside down-torso or leg, use less hand gravity
        # inside torso: probably hand lying on torso
        inside_torso = polygon_method(torch.cat([gt_joints_2d[:, [2, 1, 3, 9, 10, 4]],
                                                torch.zeros([batch_size, 6, 1], device=self.device, dtype=self.dtype)], dim=-1),
                                    torch.cat([gt_joints_2d[:, [7, 8]],
                                                torch.zeros([batch_size, 2, 1], device=self.device, dtype=self.dtype)], dim=-1))
        hand_gravity_weight[inside_torso & (cosine[:, 2:4] > 0.55)] = 1 / self.beta1 * self.beta_inside_torso
        # inside neck or leg: probably hand moving through neck or leg
        # inside_neck = polygon_method(torch.cat([gt_joints_2d[:, [2, 1, 3, 4]],
        #                                          torch.zeros([batch_size, 4, 1], device=device, dtype=dtype)], dim=-1),
        #                               torch.cat([gt_joints_2d[:, [7, 8]],
        #                                          torch.zeros([batch_size, 2, 1], device=device, dtype=dtype)], dim=-1))
        # hand_gravity_weight[inside_neck & (cosine[:, 2:4] > 0.55)] = 0
        # hand_gravity_err[inside_torso & (cosine[:, 2:4] > 0.55)] = \
        #     hand_gravity_err[inside_torso & (cosine[:, 2:4] > 0.55)] - 0.2
        inside_leg = polygon_method(torch.cat([gt_joints_2d[:, [9, 11, 12, 10]],
                                            torch.zeros([batch_size, 4, 1], device=self.device, dtype=self.dtype)], dim=-1),
                                    torch.cat([gt_joints_2d[:, [7, 8]],
                                            torch.zeros([batch_size, 2, 1], device=self.device, dtype=self.dtype)], dim=-1))
        hand_gravity_weight[inside_leg & (cosine[:, 2:4] > 0.55)] = 0
        # especially, when sitting up, the gravity constraint should be modified, i.e. use less hand gravity when human
        # sits up with arm paralleling to bed
        torso_cosine = torch.norm(gt_joints_2d[:, [3, 4]] - gt_joints_2d[:, [9, 10]], dim=-1) / torch.tensor(
            [self.dist_dict['ltorso'], self.dist_dict['rtorso']], device=self.device, dtype=self.dtype)
        sitting = ((torso_cosine[:, 0] < 0.8) & (torso_cosine[:, 1] < 0.8)).unsqueeze(1).repeat(1, 2)
        hand_gravity_weight[sitting & (cosine[:, 2:4] > 0.62)] = 1 / self.beta1 / 2

        # speed mask
        limb_vel_mask = ~vel_mask[:, [5, 6, 7, 8, 11, 12, 13, 14]]

        # bed contact mask
        bed_contact_mask = model_joints[:, [5, 6, 7, 8, 11, 12, 13, 14], -1] < 0

        # gravity loss
        gravity_err = torch.cat([elbow_gravity_err * elbow_gravity_weight,
                                hand_gravity_err * hand_gravity_weight,
                                knee_gravity_err * knee_gravity_weight,
                                feet_gravity_err * feet_gravity_weight], dim=1)
        gravity_loss1 = torch.exp(gravity_err) * limb_vel_mask * bed_contact_mask
        # adjust constant weight, additional constant weight on hand
        hand_weight = torch.ones([batch_size, 2], device=self.device, dtype=self.dtype, requires_grad=False) * 2
        hand_weight[(inside_leg) & (cosine[:, 2:4] > 0.55)] = 1
        hand_weight[(inside_torso) & (cosine[:, 2:4] > 0.55)] = self.alpha_inside_torso
        hand_weight[sitting & (cosine[:, 2:4] > 0.6)] = 0.6
        gravity_loss = (gravity_loss1[:, [2, 3]] * hand_weight).sum() + gravity_loss1[:, [0, 1, 4, 5, 6, 7]].sum()
        return gravity_loss
        pass
        