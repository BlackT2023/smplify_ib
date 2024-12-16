import numpy as np
import smplx
from tqdm import tqdm
import time
from .prior import MaxMixturePrior
from .losses import MotionOptLoss
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
                 weight=None,
                 model_type='smpl',
                 sc_type='ours',
                 dtype=torch.float32,
                 device=torch.device('cuda'),
                 ):
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
            self.geomask = self.modified_geodistssmpl < weight['geothres']
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
            self.geomask = self.geodistssmpl > weight['geothres']
        else:
            print('unknown sc type, exit')
            exit(0)
        
        # parameters for gravity
        self.bed_depth = bed_depth
        
        # necessarily observed parameters
        self.dist_dict = None
        self.gt_vel_mask = None
        self.prev_result = None

        self.gravity_fitting_loss = 
        

    def fit(self, loader, stage='gravity', sc_type='ours', fit_min=0, fit_max=10000):
        prev_result = None
        for batch in loader:
            batch = {k: v.type(torch.float32).squeeze().detach().to(self.device).requires_grad_(False) for k, v in
                        batch.items()}
            idx = batch['idx'].cpu().numpy().astype(np.int64).tolist()
            if idx[0] < fit_min:
                continue
                pass
            keypoints_2d = batch['keypoints_pix']
            init_pose = batch['est_pose']
            init_betas = batch['est_betas']
            init_trans = batch['est_trans']
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                    f', {idx[0]} - {idx[1]} start')
            if prev_result is not None:
                # print(smplify.prev_result)
                pass
            output, loss_dict = self.


        num_steps_total = len(self.loader.dataset) // self.args.batch_size

        for step, batch in enumerate(tqdm(self.loader, desc='Epoch ' + str(1),
                                          total=num_steps_total)):

            batch = {k: v.type(torch.float32).squeeze().detach().to(self.device).requires_grad_(False) for k, v in
                     batch.items()}

            output, loss_dict = self.vae_opt(
                batch
            )

            self.batch_evaluate(batch['curr_frame_idx'], batch['gt_keypoints_3d'], output['joints'], batch['verts'], output['verts'])

        # import pdb;pdb.set_trace()
        self.accel_evaluate()
        self.print_metric()


        np.save(f'results/{self.args.note}_record.npy', self.results_record)
        np.save(f'results/{self.args.note}_joints.npy', self.joints_record)

        return np.mean(self.results_record[:, 0])


    def gravity_opt(self, batch):
        """Preform gravity fitting
        Args:
            batch (dict): {'init_betas', 'init_trans', 'init_pose', 'gt_keypoints_2d', 'idx'},
        Return:
            output(dict): {'betas', 'pose', 'trans'},
            loss_dict(dict)
        """
        gt_joints_2d = batch['gt_keypoints_2d'][:, :, :2].clone()
        joints_conf = batch['gt_keypoints_2d'][:, :, -1].clone()
        
        camera_translation = batch['init_trans'].clone()
        body_pose_1 = batch['init_pose'][:, 3:30].detach().clone()
        body_pose_foot = batch['init_pose'][:, 30:36].detach().clone()
        body_pose_2 = batch['init_pose'][:, 36:].detach().clone()
        global_orient = batch['init_pose'][:, :3].detach().clone()
        betas = batch['init_betas'].detach().clone()
        
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
            
        


    def vae_opt(self, batch):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        loss_grad_record = {
            'stage 1st': [],
            'stage 2nd': []
        }

        # Split SMPL pose to body pose and global orientation
        body_pose = batch['est_pose'][:, :].detach().clone()
        betas = batch['est_betas'].detach().clone()
        transl = batch['est_trans'].detach().clone()

        # stage 1, optimize human-bed contact: translation (depth), betas, pose,
        # fix camera_translation
        transl.requires_grad = False
        body_pose.requires_grad = True
        betas.requires_grad = False

        contact_opt_params = [body_pose]

        # body_contact_optimizer = torch.optim.Adam(contact_opt_params, lr=self.step_size)
        optimizer, _ = create_optimizer(contact_opt_params, optim_type='adam', lr=self.step_size,
                                     maxiters=self.num_iters)

        # accel = compute_accel(batch['gt_keypoints_3d'].detach().cpu().numpy())

        loss_record = {}

        # if accel.max() > 0.02:
        if True:
            for i in range(self.num_iters):

                # import pdb;
                # pdb.set_trace()
                # with torch.no_grad():
                # import pdb;pdb.set_trace()
                vae_pred = self.model(body_pose)
                vae_pred = vae_pred[0]

                opt_smpl_output = self.smpl(global_orient=body_pose[:, :3],
                                        body_pose=body_pose[:, 3:],
                                        betas=betas,
                                        transl=transl
                                       )
                opt_joints = opt_smpl_output.joints[:, :24]
                pihmr_joints = self.smpl(global_orient=batch['est_pose'][:, :3],
                                        body_pose=batch['est_pose'][:, 3:],
                                        betas=betas,
                                        transl=transl
                                         ).joints[:, :24]
                vae_joints = self.smpl(global_orient=vae_pred[:, :3],
                                        body_pose=vae_pred[:, 3:],
                                        betas=betas,
                                        transl=transl).joints[:, :24]

                loss, loss_record = self.loss.vae_prior_loss(
                    body_pose,
                    vae_pred,
                    [opt_joints, pihmr_joints, vae_joints],
                    batch
                )

                loss_pre = self.loss.opt_smooth_loss(
                    body_pose,
                    opt_joints,
                    batch,
                )

                loss = loss + loss_pre

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # import pdb;pdb.set_trace()
                # print(
                #     "[Validating] Time: {} Epoch: [{}/{}]  {}".format(
                #         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                #         i,
                #         self.num_iters,
                #         print_loss(loss.item(), loss_record),
                #     ))

            # import pdb;pdb.set_trace()

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=body_pose[:, :3],
                                    body_pose=body_pose[:, 3:],
                                    betas=betas,
                                    return_full_pose=True,
                                    transl=transl)

        final_result = {
            'pose': body_pose.detach(),
            'betas': betas.detach(),
            'trans': transl.detach(),
            'joints': smpl_output.joints[:, :24].detach(),
            'verts': smpl_output.vertices.detach(),
            'faces': self.smpl.faces
        }

        # print(compute_accel(batch['gt_keypoints_3d'][:, :24].detach().cpu().numpy()).mean(), compute_accel(final_result['joints'].cpu().numpy()).mean())

        return final_result, loss_record

    def accel_evaluate(self):

        segments = self.loader.dataset.segments
        joints = self.joints_record

        accel = []

        # import pdb;pdb.set_trace()

        for segment in segments:
            accel.extend(compute_accel(joints[segment[0]:segment[1]]).tolist())


        self.results_record[:len(accel), 4] = accel

    def batch_evaluate(self, index, target_j3ds, pred_j3ds, gt_verts, pred_verts):

        # reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        #
        # # import pdb;
        # # pdb.set_trace()
        #
        # gt_verts = reduce(gt_verts)
        # target_j3ds = reduce(target_j3ds)
        # pred_verts = reduce(pred_verts)
        # pred_j3ds = reduce(pred_j3ds)

        # target_j3ds = target_j3ds[:, 0]

        index = index.reshape(-1).type(torch.long).cpu()

        errors_wo_align = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1) * 1000

        pred_pelvis = (pred_j3ds[:, [1], :] + pred_j3ds[:, [2], :]) / 2.0
        target_pelvis = (target_j3ds[:, [1], :] + target_j3ds[:, [2], :]) / 2.0

        pred_j3ds_p = pred_j3ds - pred_pelvis
        target_j3ds_p = target_j3ds - target_pelvis
        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds_p - target_j3ds_p) ** 2).sum(dim=-1)).mean(dim=-1)
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds_p, target_j3ds_p)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds_p) ** 2).sum(dim=-1)).mean(dim=-1)

        m2mm = 1000

        pve = compute_error_verts(target_verts=gt_verts, pred_verts=pred_verts) * m2mm
        mpjpe = errors * m2mm
        pa_mpjpe = errors_pa * m2mm

        # import pdb;pdb.set_trace()
        self.results_record[index, :4] = torch.stack([
            errors_wo_align, mpjpe, pa_mpjpe, pve
        ], dim=1).detach().cpu().numpy()
        self.joints_record[index] = pred_j3ds.detach().cpu().numpy()

    def print_metric(self):
        segments = len(self.loader.dataset.segments)

        loss_dict = {
            'mpjpe_wo_align': np.mean(self.results_record[:, 0]),
            'mpjpe': np.mean(self.results_record[:, 1]),
            'mpjpe_pa': np.mean(self.results_record[:, 2]),
            'mpve': np.mean(self.results_record[:, 3]),
            'accel': np.mean(self.results_record[:-segments * 2, 4]) * 1000,
            # 'accel_error': np.mean(self.val_results_record[:-segments * 2, 5])
        }

        print(
            "[Validating] Time: {}, {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                print_loss(0, loss_dict),
            ))


        return loss_dict
