import numpy as np
import torch
from torch.utils.data import DataLoader
import smplx
from tqdm import tqdm
import time
from config import config

from data.essentials.segments.smpl import segm_utils as exn
from utils.optimizers.optim_factory import create_optimizer
from utils.geometry.camera import PerspectiveCamera
from utils.contact.Segmentation import BatchBodySegment
from SMPLify.losses import GravityFittingLoss, SelfcontactFittingLoss, CameraFittingLoss

from core.evaluate import *
from datetime import datetime

class SMPLifyIB():
    def __init__(self,
                 step_size=1e-2,
                 num_iters=300,
                 batch_size=128,
                 bed_depth=1.66,
                 weights=None,
                 model_type='smpl',
                 bc_type='simple',
                 sc_type='ours',
                 dtype=torch.float32,
                 device=torch.device('cuda')):
        self.device = device
        self.dtype = dtype
        
        self.step_size = step_size
        self.num_iters = num_iters
        self.batch_size = batch_size
        
        # camera
        camera = PerspectiveCamera(focal_length_x=941,
                                   focal_length_y=941,
                                   batch_size=batch_size,
                                   center=torch.Tensor([1080 // 2, 1920 // 2]),
                                   dtype=dtype)
        self.camera = camera.to(device=device)

        self.body_model = smplx.create(config.SMPL_MODEL_DIR, model_type=model_type, gender='neutral').to(self.device).eval()
        
        self.faces = self.body_model.faces
        self.face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                        device=self.device).unsqueeze_(0).repeat([128, 1, 1])
        self.segments = BatchBodySegment([x for x in exn.segments.keys()], self.face_tensor[0])
        
        # parameters for self-contact
        geodistssmpl = torch.tensor(np.load(config.SMPL_GEODIST), dtype=self.dtype, device=self.device)
        self.modified_geodistssmpl, self.geomask, self.search_tree, self.pen_distance, self.tuch_segment = None, None, None, None, None
        if sc_type == 'ours':
            from utils.contact.Contact import modify_geodistsmpl
            self.modified_geodistssmpl = modify_geodistsmpl(geodistssmpl, self.segments)
            self.modified_mask = self.modified_geodistssmpl < weights['geothres']
        elif sc_type == 'bvh':
            from mesh_intersection.bvh_search_tree import BVH
            import mesh_intersection.loss as collisions_loss
            from mesh_intersection.filter_faces import FilterFaces
            import pickle
            
            ign_part_pairs = ['9,16', '9,17', '6,16', '6,17', '1,2', '12,22']
            max_collisions = 128
            df_cone_height = 0.0001
            penalize_outside = True
            point2plane = False

            search_tree = BVH(max_collisions=max_collisions)

            pen_distance = \
                collisions_loss.DistanceFieldPenetrationLoss(
                    sigma=df_cone_height, point2plane=point2plane,
                    vectorized=True, penalize_outside=penalize_outside)

            self.search_tree = search_tree
            self.pen_distance = pen_distance
        elif sc_type == 'tuch':
            from utils.ori_tuch.Segmentation import BatchBodySegment as tuch_BatchBodySegment
            self.tuch_segment = tuch_BatchBodySegment([x for x in exn.segments.keys()], self.face_tensor[0])
            self.geodistssmpl = geodistssmpl
            self.geomask = self.geodistssmpl > weights['geothres']
        else:
            print('unknown sc type, exit')
            exit(0)
        
        # parameters for gravity
        self.bed_depth = bed_depth
        
        # necessarily observed parameters
        self.dataset = None
        self.dist_dict = None
        self.gt_vel_mask = None

        # load weights
        camera_fitting_weights, gravity_fitting_weights, sc_fitting_weights = {}, {}, {}
        for key, value in weights.items():
            camera_fitting_weights[key] = value[0]
            gravity_fitting_weights[key] = value[1]
            sc_fitting_weights[key] = value[2]
        self.camera_fitting_loss = CameraFittingLoss(weights=camera_fitting_weights,
                                                     camera=self.camera,
                                                     segments=self.segments,
                                                     bed_depth=self.bed_depth,
                                                     device=self.device,
                                                     dtype=self.dtype)
        self.gravity_fitting_loss = GravityFittingLoss(weights=gravity_fitting_weights,
                                                       camera=self.camera,
                                                       dist_dict=None,
                                                       bc_type=bc_type,
                                                       sc_type=sc_type,
                                                       bed_depth=self.bed_depth,
                                                       segments=self.segments,
                                                       modified_mask=self.modified_mask,
                                                       geomask=self.geomask,
                                                       faces=self.face_tensor,
                                                       search_tree=self.search_tree,
                                                       pen_distance=self.pen_distance,
                                                       device=self.device,
                                                       dtype=self.dtype)
        self.sc_fitting_loss = SelfcontactFittingLoss(weights=sc_fitting_weights,
                                                       camera=self.camera,
                                                       dist_dict=None,
                                                       bc_type=bc_type,
                                                       sc_type=sc_type,
                                                       bed_depth=self.bed_depth,
                                                       segments=self.segments,
                                                       modified_mask=self.modified_mask,
                                                       geomask=self.geomask,
                                                       faces=self.face_tensor,
                                                       search_tree=self.search_tree,
                                                       pen_distance=self.pen_distance,
                                                       device=self.device,
                                                       dtype=self.dtype)
        

    def fit(self, dataset, stage='gravity', sc_type='ours', vel_thres=110, fit_min=0, fit_max=10000):
        self.dataset = dataset
        self.pre_process(vel_thre=vel_thres)
        loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)
        length = loader.betas.shape[0]
        opt_betas = np.zeros([length, 10], dtype=np.float32)
        opt_transl = np.zeros([length, 3], dtype=np.float32)
        opt_pose = np.zeros([length, 72], dtype=np.float32)

        prev_result = None
        for batch in loader:
            batch = {k: v.type(torch.float32).squeeze().detach().to(self.device).requires_grad_(False) for k, v in
                        batch.items()}
            idx = batch['idx'].cpu().numpy().astype(np.int64).tolist()
            if idx[0] < fit_min:
                continue
                pass
            opt_batch = self.prevent_overturn(init_pose=batch['est_pose'],
                                              init_betas=batch['est_betas'],
                                              init_trans=batch['est_trans'],
                                              gt_keypoints=batch['keypoints_pix'])
            opt_batch['idx'] = idx

            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                    f', {idx[0]} - {idx[1]} start')
            if prev_result is not None:
                # print(smplify.prev_result)
                pass
            if stage == 'gravity':
                output, loss_dict = self.gravity_opt(batch)
            elif stage == 'contact':
                output, loss_dict = self.contact_opt(batch)
            else:
                print('unknown stage, exit 0')
                exit(0)
            opt_betas[idx[0]:idx[1]] = output['betas']
            opt_transl[idx[0]:idx[1]] = output['transl']
            opt_pose[idx[0]:idx[1]] = output['pose']
        return opt_betas, opt_transl, opt_pose


    def gravity_opt(self, batch):
        """Preform camera fitting and gravity fitting
        Args:
            batch (dict): {'init_betas', 'init_trans', 'init_pose', 'gt_keypoints_2d', 'idx'},
        Return:
            output(dict): {'betas', 'pose', 'trans'},
            loss_dict(dict)
        """
        gt_joints_2d = batch['gt_keypoints_2d'][:, :, :2].clone()
        joints_conf = batch['gt_keypoints_2d'][:, :, -1].clone()
        idx = batch['idx']
        
        camera_translation = batch['init_trans'].clone()
        body_pose_1 = batch['init_pose'][:, 3:30].detach().clone()
        body_pose_foot = batch['init_pose'][:, 30:36].detach().clone()
        body_pose_2 = batch['init_pose'][:, 36:].detach().clone()
        global_orient = batch['init_pose'][:, :3].detach().clone()
        betas = batch['init_betas'].detach().clone()

        prev_result = None
        
        # Step 0, camera fitting
        camera_translation.requires_grad = True
        body_pose_1.requires_grad = False
        body_pose_foot.requires_grad = False
        body_pose_2.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = False
        
        camera_opt_params = [camera_translation]
        camera_optimizer, _ = create_optimizer(camera_opt_params,
                                               optim_type=self.optim_type,
                                               lr=self.step_size,
                                               maxiters=50)
        for i in range(80):
            smpl_output = self.body_model(global_orient=global_orient,
                                          body_pose=torch.cat([body_pose_1, body_pose_foot, body_pose_2], dim=1),
                                          betas=betas,
                                          transl=camera_translation)
            reproj_joints = smpl_output.joints[:, config.reproj_idx] + torch.tensor([0, 0, self.bed_depth], device=self.device, dtype=self.dtype)
            vertices = smpl_output.vertices + torch.tensor([0, 0, self.bed_depth], device=self.device, dtype=self.dtype)
            camera_loss = self.camera_fitting_loss(global_orient=global_orient,
                                                   body_pose=torch.cat([body_pose_1, body_pose_foot, body_pose_2], dim=1),
                                                   betas=betas,
                                                   transl=camera_translation,
                                                   joints=reproj_joints,
                                                   vertices=vertices,
                                                   vel_mask=self.gt_vel_mask[idx[0]:idx[1]],
                                                   gt_keypoints_2d=gt_joints_2d,
                                                   gt_keypoints_conf=joints_conf,
                                                   prev_result=prev_result,
                                                   idx=idx)
            camera_optimizer.zero_grad()
            camera_loss.backward()
            camera_optimizer.step()
        
        # Step 1, gravity fitting
        camera_translation.requires_grad = True
        body_pose_1.requires_grad = True
        body_pose_foot.requires_grad = False
        body_pose_2.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True

        depth_opt_params = [body_pose_1, body_pose_2, betas, global_orient, camera_translation]
        depth_optimizer, _ = create_optimizer(depth_opt_params,
                                              optim_type=self.optim_type,
                                              lr=self.step_size,
                                              maxiters=30)
        
        for i in range(self.num_iters):
            smpl_output = self.body_model(global_orient=global_orient,
                                          body_pose=torch.cat([body_pose_1, body_pose_foot, body_pose_2], dim=1),
                                          betas=betas,
                                          transl=camera_translation)
            reproj_joints = smpl_output.joints[:, config.reproj_idx] + torch.tensor([0, 0, self.bed_depth], device=self.device, dtype=self.dtype)
            smooth_joints = smpl_output.joints[:, config.smooth_idx] + torch.tensor([0, 0, self.bed_depth], device=self.device, dtype=self.dtype)
            vertices = smpl_output.vertices + torch.tensor([0, 0, self.bed_depth], device=self.device, dtype=self.dtype)
            

            
    def contact_opt(self, batch):
        """Preform self-contact fitting
        Args:
            batch (dict): {'init_betas', 'init_trans', 'init_pose', 'gt_keypoints_2d', 'idx'},
        Return:
            output(dict): {'betas', 'pose', 'trans'},
            loss_dict(dict)
        """
        pass


    def batch_evaluate(self):
        pass

    
    def pre_process(self, vel_thre):
        # get gt_vel_mask, dist_dict of one dataset
        gt_kp = torch.from_numpy(self.dataset.keypoints_pix[:, :, :2], device=self.device, dtype=self.dtype)
        gt_vel = ((gt_kp[:-1] - gt_kp[1:]) ** 2).sum(-1)
        gt_vel = torch.cat((gt_vel, gt_vel[-1].clone().unsqueeze(0)), dim=0)
        self.gt_vel_mask = (gt_vel > vel_thre)

        f0 = gt_kp[0]
        self.dist_dict = {
            'elbow': torch.mean(torch.norm(f0[[3, 4], :2] - f0[[5, 6], :2], dim=-1), dim=-1),
            'hand': torch.mean(torch.norm(f0[[5, 6], :2] - f0[[7, 8], :2], dim=-1), dim=-1),
            'knee': torch.mean(torch.norm(f0[[9, 10], :2] - f0[[11, 12], :2], dim=-1), dim=-1),
            'feet': torch.mean(torch.norm(f0[[11, 12], :2] - f0[[13, 14], :2], dim=-1), dim=-1),
            'ltorso': torch.norm(f0[3, :2] - f0[9, :2], dim=-1),
            'rtorso': torch.norm(f0[4, :2] - f0[10, :2], dim=-1),
            'shoulder': torch.norm(f0[3, :2] - f0[4, :2], dim=-1)
        }

        
    def prevent_overturn(self,
                         init_pose,
                         init_betas,
                         init_trans,
                         gt_keypoints):
        smpl_output = self.smpl(
            global_orient=init_pose[:, :3],
            body_pose=init_pose[:, 3:],
            betas=init_betas,
            trans=init_trans
        )
        joints_3d = smpl_output.joints[:, config.reproj_idx] + torch.tensor([0, 0, self.bed_depth]).to(self.device)
        joints_2d = self.camera(joints_3d)
        gt_keypoints_2d = gt_keypoints[:, :, :2]

        l2rshoulder = joints_2d[:, 3] - joints_2d[:, 4]
        l2rshoulder_gt = gt_keypoints_2d[:, 3] - gt_keypoints_2d[:, 4]

        direction = (l2rshoulder * l2rshoulder_gt).sum(-1) / torch.norm(l2rshoulder, dim=-1) / torch.norm(
            l2rshoulder_gt, dim=-1)
        cosine = torch.norm(l2rshoulder_gt, dim=-1) / self.dist_dict['shoulder']

        reverse = (cosine > 0.5) & (direction < -0.8)
        if reverse[0]:
            init_pose[0], init_betas[0], init_trans[0] = init_pose[1].clone(), init_betas[1].clone(), init_trans[
                0].clone()
        reverse[0] = False
        reverse_idx = torch.nonzero(reverse).squeeze(-1)
        for idx in reverse_idx:
            init_pose[idx] = init_pose[idx - 1].clone()
            init_betas[idx] = init_betas[idx - 1].clone()
            init_trans[idx] = init_trans[idx - 1].clone()
        return {
            'init_pose': init_pose,
            'init_betas': init_betas,
            'init_trans': init_trans,
            'gt_keypoints_2d': gt_keypoints
        }


