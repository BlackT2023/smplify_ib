# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# these codes are from https://github.com/muelea/tuch


import sys
import torch
import trimesh
import torch.nn as nn
import numpy as np
import os.path as osp
from config import config

from data.essentials.segments.smpl import segm_utils as exn


class BodySegment(nn.Module):
    def __init__(self,
                 name,
                 faces,
                 append_idx=None):
        super(BodySegment, self).__init__()
        self.device = faces.device
        self.name = name
        self.append_idx = faces.max().item() if append_idx is None else append_idx

        # read mesh and find faces of segment
        path = osp.join(config.SEGMENT_DIR, 'smpl_segment_{}.ply'.format(name))
        bandmesh = trimesh.load(path, process=False)
        self.segment_vidx = np.where(np.array(bandmesh.visual.vertex_colors[:, 0]) == 255)[0]

        # recreate an index map
        included_verts = np.isin(np.arange(self.append_idx + 1), self.segment_vidx)
        verts_idx_map = np.cumsum(included_verts.astype(int)) - 1

        # read boundary information
        # bands_faces is mapped
        self.bands = [x for x in exn.segments[name].keys()]
        self.bands_verts = [x for x in exn.segments[name].values()]
        self.bands_faces = self.create_band_faces(verts_idx_map).to(self.device)

        # read mesh and find, apply index map to segment_faces
        # only bands_faces, segment_faces is mapped
        faces = faces.squeeze()
        segment_faces_ids = np.where(np.isin(faces.cpu().numpy(), self.segment_vidx).sum(1) == 3)[0]
        segment_faces = faces[segment_faces_ids, :]
        shape = segment_faces.shape
        mapped_flat_segment_faces = torch.index_select(torch.from_numpy(verts_idx_map).to(self.device),
                                                       0,
                                                       segment_faces.view(-1))
        mapped_segment_faces = mapped_flat_segment_faces.view(shape)
        mapped_segment_faces = torch.cat((mapped_segment_faces, self.bands_faces), 0)
        self.register_buffer('mapped_segment_faces', mapped_segment_faces)

    def create_band_faces(self, verts_idx_map):
        """
            create the faces that close the segment.
        """
        bands_faces = []
        for idx, k in enumerate(self.bands):
            new_vert_idx = verts_idx_map[-1] + 1 + idx
            new_faces = [[
                verts_idx_map[self.bands_verts[idx][i + 1]],
                verts_idx_map[self.bands_verts[idx][i]],
                new_vert_idx] \
                for i in range(len(self.bands_verts[idx]) - 1)]
            bands_faces += new_faces
        return torch.tensor(np.array(bands_faces).astype(np.int64), dtype=torch.long)

    def get_closed_segment(self, vertices):
        """
            create the closed segment mesh from SMPL-X vertices.
        """
        segm_verts = vertices[:, self.segment_vidx, :]
        # append vertices to SMPLX vertices, that close the segment and compute faces
        for bv in self.bands_verts:
            close_segment_vertices = torch.mean(vertices[:, bv, :], 1, keepdim=True)
            segm_verts = torch.cat((segm_verts, close_segment_vertices), 1)
        # the 557th point of head segment is wrong
        if self.name == 'head':
            segm_verts = torch.cat((segm_verts[:, 0:557, :],
                                    segm_verts[:, 556, :].unsqueeze(1),
                                    segm_verts[:, 558:, :]),
                                   dim=1)
        return segm_verts, self.mapped_segment_faces

    def get_unclosed_segment_verts(self, vertices):
        segm_verts = vertices[:, self.segment_vidx, :]
        # the 557th point of head segment is wrong
        if self.name == 'head':
            segm_verts = torch.cat((segm_verts[:, 0:557, :],
                                    segm_verts[:, 556, :].unsqueeze(1),
                                    segm_verts[:, 558:, :]),
                                   dim=1)
        return segm_verts

    def get_segment_vidx(self):
        return self.segment_vidx


class BatchBodySegment(nn.Module):
    def __init__(self,
                 names,
                 faces,
                 nearest_center=None,
                 device=torch.device('cuda')):
        super(BatchBodySegment, self).__init__()
        self.names = names
        self.nv = faces.max().item()
        self.segmentation = {}
        for idx, name in enumerate(names):
            self.segmentation[name] = BodySegment(name, faces)

        if nearest_center is None:
        # if True:
            self.has_nearest_center = False
            nearest_center = self.cal_nearest_center()
            # np.savez(config.NEAREST_CENTER, **nearest_center)
            import pickle
            with open('data/nearest_center.pkl', 'wb') as file:
                pickle.dump(nearest_center, file)
            for key in nearest_center.keys():
                nearest_center[key] = torch.from_numpy(nearest_center[key]).to(device)
            self.nearest_center = nearest_center
            self.has_nearest_center = True
        else:
            self.nearest_center = nearest_center
            self.has_nearest_center = True

    def get_unclosed_segment(self, vertices, name, smooth_joints=None, requires_center=False):
        verts = self.segmentation[name].get_unclosed_segment_verts(vertices)
        if requires_center:
            if smooth_joints is None:
                print('requires_center is True, while smooth_joints is None')
                return None
            # find joints which are inside the segment
            # arm -> elbow, wrist
            # thigh -> knee, ankle
            # head -> nose, lear, rear
            # torse_upperarm -> lshoulder, rshoulder
            # torse_thigh -> lhip, rhip
            if name == 'left_arm':
                centers = torch.cat([smooth_joints[:, [5, 7]],
                                     (smooth_joints[:, 5] + smooth_joints[:, 7]).unsqueeze(1) / 2,
                                     (3 * smooth_joints[:, 5] + 2 * smooth_joints[:, 3]).unsqueeze(1) / 5],
                                    dim=1)
            elif name == 'right_arm':
                centers = torch.cat([smooth_joints[:, [6, 8]],
                                     (smooth_joints[:, 6] + smooth_joints[:, 8]).unsqueeze(1) / 2,
                                     (3 * smooth_joints[:, 6] + 2 * smooth_joints[:, 4]).unsqueeze(1) / 5],
                                    dim=1)
            elif name == 'left_thigh':
                centers = torch.cat([smooth_joints[:, [11, 13]],
                                     (smooth_joints[:, 11] + smooth_joints[:, 13]).unsqueeze(1) / 2,
                                     (3 * smooth_joints[:, 11] + 2 * smooth_joints[:, 9]).unsqueeze(1) / 5],
                                    dim=1)
            elif name == 'right_thigh':
                centers = torch.cat([smooth_joints[:, [12, 14]],
                                     (smooth_joints[:, 12] + smooth_joints[:, 14]).unsqueeze(1) / 2,
                                     (3 * smooth_joints[:, 12] + 2 * smooth_joints[:, 10]).unsqueeze(1) / 5],
                                    dim=1)
            elif name == 'head':
                centers = smooth_joints[:, [0, 1, 2]]
            elif name == 'torso_thigh':
                centers = smooth_joints[:, [9, 10]]
            elif name == 'torso_upperarm':
                centers = smooth_joints[:, [3, 4, 15]]
            else:
                print('unknown segment name: ', name)
                return None
            # for self.cal_nearest_center
            if self.has_nearest_center:
                return verts, centers, self.nearest_center[name]
            else:
                return verts, centers, None
        else:
            return verts, None, None

    def get_unclosed_segments(self, vertices, smooth_joints, names, requires_center=False):
        device = vertices.device
        dtype = vertices.dtype
        batch_size, _, dim = vertices.shape
        verts = torch.empty([batch_size, 0, dim], dtype=dtype, device=device, requires_grad=True)
        if requires_center:
            centers = torch.empty([batch_size, 0, dim], dtype=dtype, device=device, requires_grad=True)
            nearest_center = torch.empty([0], dtype=torch.int64, device=device, requires_grad=False)
            for name in names:
                seg_verts, seg_centers, seg_nearest_center = \
                    self.get_unclosed_segment(vertices=vertices,
                                              name=name,
                                              smooth_joints=smooth_joints,
                                              requires_center=requires_center)
                verts = torch.cat([verts, seg_verts], dim=1)
                nearest_center = torch.cat([nearest_center, seg_nearest_center + centers.shape[1]], dim=0)
                centers = torch.cat([centers, seg_centers], dim=1)
            return verts, centers, nearest_center
        else:
            for name in names:
                seg_verts = self.segmentation[name].get_unclosed_segment_verts(vertices)
                verts = torch.cat([verts, seg_verts], dim=1)
            return verts, None, None

    def get_segment_vidx(self, name):
        return self.segmentation[name].get_segment_vidx()

    def get_segments_vidx(self, names):
        if len(names) == 0:
            print('No segments found')
            return None
        ret = np.empty([0], dtype=int)
        for name in names:
            ret = np.concatenate((ret, self.segmentation[name].get_segment_vidx()))
        return ret

    def cal_nearest_center(self):
        dtype = torch.float32
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # rest smpl pose
        import smplx

        model_type = 'smpl'
        body_model = smplx.create(
            'data/models',
            model_type=model_type,
            gender='neutral'
        ).to(device)
        body_model.eval()

        batch_size = 8
        pose = torch.zeros([batch_size, 72], dtype=dtype, device=device)
        betas = torch.zeros([batch_size, 10], dtype=dtype, device=device)
        trans = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        smpl_output = body_model(global_orient=pose[:, :3],
                                 body_pose=pose[:, 3:],
                                 betas=betas,
                                 transl=trans)
        vertices = smpl_output.vertices[0].unsqueeze(0)
        center_joints = smpl_output.joints[0, config.smooth_idx].unsqueeze(0)
        print(center_joints)
        # exit()

        # nearset center
        from utils.advance_contact.myContact import batch_knn
        all_segs = ['head', 'torso_thigh', 'torso_upperarm',
                    'left_arm', 'right_arm', 'left_thigh', 'right_thigh']
        nearest_center = dict()
        for seg in all_segs:
            db, center, _ = self.get_unclosed_segment(vertices=vertices,
                                                   smooth_joints=center_joints,
                                                   name=seg,
                                                   requires_center=True)
            center_idx = batch_knn(center, db)
            nearest_center[seg] = center_idx[0].cpu().numpy()
        return nearest_center
