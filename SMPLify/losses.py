import setup
import torch
import torch.nn as nn
import pickle
import numpy as np

from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss
from utils.geometry.geometry import batch_rodrigues

from pytorch3d.renderer import (
                                RasterizationSettings, MeshRenderer,
                                MeshRasterizer,
                                SoftSilhouetteShader,
                                )
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments

from pytorch3d.structures import Meshes

def gmof(x, sigma):
    """
    Geman-McClure error function
    https://github.com/nkolot/SPIN/blob/master/smplify/losses.py
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

class SelfShader(nn.Module):
    """
    Calculate the silhouette by blending the top K faces for each pixel based
    on the 2d euclidean distance of the center of the pixel to the mesh face.
    color means the z_dis from pixel to faces
    Use this shader for generating silhouettes similar to SoftRasterizer [0].

    .. note::
    """

    def __init__(self, blend_params=None) -> None:
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        """
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        # colors = torch.ones_like(fragments.bary_coords)
        colors = fragments.zbuf.clone().unsqueeze(4).repeat(1, 1, 1, 1, 3)

        blend_params = kwargs.get("blend_params", self.blend_params)
        images = sigmoid_alpha_blend(colors, fragments, blend_params)
        return images

class MotionOptLoss(nn.Module):
    def __init__(
            self,
            args,
            faces,
            balance_weight=0.8,
            smpl_loss_weight=1.,
            joint_loss_weight=1.,
            smooth_weight=1.,
            pre_smooth_weight=10.,
            contact_loss_weight=0.001,
            device='cuda',
    ):
        super(MotionOptLoss, self).__init__()
        self.args = args
        self.faces = faces
        self.balance_weight = balance_weight
        self.smpl_loss_weight = smpl_loss_weight
        self.joint_loss_weight = joint_loss_weight
        self.smooth_weight = smooth_weight
        self.pre_smooth_weight = pre_smooth_weight
        self.contact_loss_weight = contact_loss_weight
        self.device = device

        # for calculate pressure iou
        self.camera_silh = OrthographicCameras(
            device=self.device,
            focal_length=torch.Tensor([[-1 / self.args.sensor_pitch[1], -1 / self.args.sensor_pitch[0]]]),
            image_size=torch.Tensor([self.args.sensor_size]),
            principal_point=torch.Tensor([[0, 0]]),
            in_ndc=False)

        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=self.args.sensor_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50,
            bin_size=0,
        )

        # self.renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(
        #     cameras=self.camera_silh, raster_settings=raster_settings_soft),
        #     shader=SoftSilhouetteShader())

        self.renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(
            cameras=self.camera_silh, raster_settings=raster_settings_soft),
            shader=SelfShader())

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

    def keypoints_3d(self, pred_keypoints_3d, gt_keypoints_3d):
        # gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        # gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        # pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        # pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
        # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()

    def smpl_losses(self, opt_pose, gt_pose):
        pred_rotmat_valid = batch_rodrigues(opt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
        return loss_regr_pose

    def pressure_projection(self, verts, gt_pressure_binary):

        meshes_world = Meshes(verts=verts[:, :, :], faces=self.faces[:, :, :])
        images_predicted = self.renderer_silhouette(meshes_world, cameras=self.camera_silh, lights=None)[:, :, :]
        depth_predicted = images_predicted[:, :, :, 0] + torch.tensor([1]).to(self.device)
        # import pdb;pdb.set_trace()
        sihoule_sumloss_value = self.sihoule_sumloss(depth_predicted, gt_pressure_binary)
        sihoule_iouloss_value = self.sihoule_iouloss(depth_predicted, gt_pressure_binary)
        ratio = 0.5
        loss_sihoule = sihoule_sumloss_value * ratio + sihoule_iouloss_value * 100 * (1 - ratio)
        return loss_sihoule.mean()

    def batch_smooth_pose_loss(self, pred_pose):
        pose_diff = pred_pose[1:] - pred_pose[:-1]
        return torch.mean(pose_diff.abs())

    def batch_smooth_joint_loss(self, pred_joint):
        pose_diff = pred_joint[1:] - pred_joint[:-1]
        return torch.mean(pose_diff.abs())

    def sihoule_sumloss(self, pred_pressure_binary, gt_pressure_binary):
        loss_fn = torch.nn.L1Loss(reduce=False)
        return loss_fn(pred_pressure_binary.sum((1, 2)), gt_pressure_binary.sum((1, 2)))

    def sihoule_iouloss(self, pred_pressure_binary, gt_pressure_binary):
        loss_fn = torch.nn.L1Loss(reduce=False)
        tmp_divide = pred_pressure_binary * gt_pressure_binary
        divide_num = tmp_divide.sum(dim=(1, 2))

        tmp_union = pred_pressure_binary + gt_pressure_binary - tmp_divide
        union_num = tmp_union.sum(dim=(1, 2))

        iou = divide_num / union_num

        return 1 - iou

    def vae_prior_loss(
            self,
            opt_est,
            vae_est,
            joints_group,
            batch,
    ):
        # import pdb;
        # pdb.set_trace()
        # pose loss
        pihmr_pose_loss = self.smpl_losses(opt_est, batch['est_pose'])
        vae_pose_loss = self.smpl_losses(opt_est, vae_est)

        # joint loss
        pihmr_joint_loss = self.keypoints_3d(joints_group[0], joints_group[1]).sum()
        vae_joint_loss = self.keypoints_3d(joints_group[0], joints_group[2]).sum()

        # smooth_loss
        pose_smooth_loss = self.batch_smooth_pose_loss(opt_est)
        joint_smooth_loss = self.batch_smooth_joint_loss(joints_group[0])

        # contact loss
        loss_press_proj = torch.tensor(0).to(self.device)
        # loss_press_proj = self.pressure_projection(opt_verts, batch['binary_pressure'])

        loss_record = {
            'pihmr_pose_loss': pihmr_pose_loss.item(),
            'vae_pose_loss': vae_pose_loss.item(),
            'pihmr_joint_loss': pihmr_joint_loss.item(),
            'vae_joint_loss': vae_joint_loss.item(),
            'pose_smooth_loss': pose_smooth_loss.item(),
            'joint_smooth_loss': joint_smooth_loss.item(),
            'sihoule_loss': loss_press_proj.item(),
        }
        # import pdb;pdb.set_trace()
        loss = self.smpl_loss_weight * (self.balance_weight * pihmr_pose_loss + (1 - self.balance_weight) * vae_pose_loss) \
            + self.joint_loss_weight * (self.balance_weight * pihmr_joint_loss + (1 - self.balance_weight) * vae_joint_loss) \
            + self.smooth_weight * (pose_smooth_loss + joint_smooth_loss) \
            + self.contact_loss_weight * loss_press_proj

        return loss, loss_record

    def opt_smooth_loss(self, opt_est, opt_joints, batch):
        loss = torch.tensor(0.).to(self.device)
        if 'pre_pose' in batch.keys():
            # import pdb;pdb.set_trace()
            loss += self.batch_smooth_pose_loss(torch.cat([batch['pre_pose'], opt_est[:16]], dim=0))
            loss += self.batch_smooth_joint_loss(torch.cat([batch['pre_joints'], opt_joints[:16]], dim=0))
        return loss * self.pre_smooth_weight



