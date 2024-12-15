import torch
from utils.contact.Contact import seg2segSDF
import time


class SelfContactLoss():
    def __init__(self,
                 segments,
                 modified_mask,
                 geomask=None,
                 faces=None,
                 search_tree=None,
                 pen_distance=None,
                 sc_euclthres=0.02,
                 sc_inside_alpha=1,
                 sc_outside_alpha=0.005,
                 sc_inside_beta=0.04,
                 sc_outside_beta=0.005,
                 sc_joint_alpha=1,
                 sc_joint_beta=1,
                 device='cuda',
                 dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        self.segments = segments
        self.modified_mask = modified_mask
        self.euclthres = sc_euclthres
        self.xyz_weight = torch.tensor([1, 1, 2], dtype=dtype, device=device)
        
        # for TUCH self-contact module
        self.geomask = geomask
        self.faces = faces
        
        # for BVH self-contact module
        self.pen_distance = pen_distance
        self.search_tree = search_tree
        
        self.inside_alpha = sc_inside_alpha
        self.outside_alpha = sc_outside_alpha
        self.inside_beta = sc_inside_beta
        self.outside_beta = sc_outside_beta
        
        self.joint_alpha = sc_joint_alpha
        self.joint_beta = sc_joint_beta


    def __call__(self,
                 seg2seg_list,
                 verts,
                 joints=None,
                 sc_type='ours',
                 sample_rate=3):
        # for ours method, seg2seg_list contains [(q_segs, d_segs), (q_segs, d_segs), ...]
        # query_segs is the list of penetrating segments
        # database_segs is the list of penetrated segments
        # the gratitude of database_segs vertices, such as torso vertices, is removed
        if sc_type == 'ours':
            sc_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device, requires_grad=True)
            for seg2seg in seg2seg_list:
                sc_loss = sc_loss + self.ours_contact_loss(verts=verts,
                                                           joints=joints,
                                                           seg2seg=seg2seg,
                                                           sample_rate=sample_rate)
            return sc_loss
            pass
        elif sc_type == 'bvh':
            return self.bvh_loss(verts)
        elif sc_type == 'tuch_ld':
            return self.tuch_ld_loss(verts)
        pass
    
    
    def ours_contact_loss(self,
                          verts,
                          joints,   
                          seg2seg,
                          sample_rate=3):
        q_segs = seg2seg[0]
        d_segs = seg2seg[1]
        q_verts, q_centers, q_nearest_center = self.segments.get_unclosed_segments(vertices=verts,
                                                                            smooth_joints=joints,
                                                                            names=q_segs,
                                                                            requires_center=True)
        d_verts, d_centers, d_nearest_center = self.segments.get_unclosed_segments(vertices=verts,
                                                                            smooth_joints=joints,
                                                                            names=d_segs,
                                                                            requires_center=True)
        q_idx, d_idx = self.segments.get_segments_vidx(q_segs), self.segments.get_segments_vidx(d_segs)
        sdf, joint_dist = seg2segSDF(db=d_verts[:, ::sample_rate].clone().detach(),
                                    queries=q_verts[:, ::sample_rate],
                                    centers=d_centers.clone().detach(),
                                    centers2=q_centers,
                                    # centers2=torch.cat([q_centers[:, :, :2].clone().detach(),
                                    #                     q_centers[:, :, 2].unsqueeze(2)], dim=-1),
                                    # queries=torch.cat([q_verts[:, ::sample_rate, :2].clone().detach(),
                                    #                     q_verts[:, ::sample_rate, 2].unsqueeze(2)], dim=-1),
                                    center_idx=d_nearest_center[::sample_rate],
                                    center_idx2=q_nearest_center[::sample_rate],
                                    geomask=self.modified_mask[q_idx[::sample_rate]][:, d_idx[::sample_rate]],
                                    output='joint_dist')
        inside = sdf < 0
        outside = (sdf > 0) & (sdf < self.euclthres)
        inside_loss = (torch.tanh(sdf[inside] / (-self.inside_beta)) ** 2).sum()
        outside_loss = (torch.tanh(sdf[outside] / self.outside_beta) ** 2).sum()
        joint_loss = joint_loss + (
                torch.exp(joint_dist[inside] / (-self.joint_beta)).sum() -
                0.01 * torch.exp(joint_dist[outside] / (-self.joint_beta)).sum()
        )
        return inside_loss * self.inside_alpha + \
               outside_loss * self.outside_alpha + \
               joint_loss * self.joint_alpha
        pass
    
    
    def tuch_ld_loss(self, verts):
        from utils.TUCH.contact import batch_pairwise_dist, winding_numbers
        batch_size = verts.shape[0]

        sc_loss = torch.tensor(0, device=self.device, dtype=self.dtype, requires_grad=True)

        start = time.time()
        for bidx in range(batch_size):
            # squared pairwise distnace between vertices
            # calculate by batch_pairwise_dist in utils.contact
            #begin = time.time()
            pred_verts_dists = batch_pairwise_dist(verts[[bidx], :, :],
                                                    verts[[bidx], :, :],
                                                    squared=True)
            #l1 = time.time()
            with torch.no_grad():
                # find intersecting vertices
                triangles = (verts[bidx])[self.faces[0]]
                exterior = winding_numbers(verts[[bidx], :, :], triangles[None]).squeeze().le(0.99)

                # filter allowed self intersections
                if self.segments is not None and (~exterior).sum() > 0:
                    test_segments = self.segments.batch_has_self_isec(verts[[bidx], :, :])
                    for segm_name, segm_ext in zip(self.segments.names, test_segments):
                        exterior[self.segments.segmentation[segm_name] \
                            .segment_vidx[(segm_ext).detach().cpu().numpy() == 0]] = 1

                # find closest vertex in contact
                pred_verts_dists[:, ~self.geomask] = float('inf')
                pred_verts_incontact_argmin = torch.argmin(pred_verts_dists, axis=1)[0]
            # general contact term to pull vertices that are close together
            pred_verts_incontact_min = torch.norm(verts[bidx] - verts[bidx, pred_verts_incontact_argmin, :], dim=1)
            in_contact = pred_verts_incontact_min < self.euclthres
            if (~exterior).sum() > 0:
                v2vinside = self.inside_alpha * torch.tanh(pred_verts_incontact_min[~exterior] / self.inside_beta) ** 2
            else:
                v2vinside = torch.tensor(0, device=self.device, dtype=self.dtype, requires_grad=True)
            ext_and_in_contact = exterior & in_contact
            if ext_and_in_contact.sum() > 0:
                v2voutside = self.outside_alpha * torch.tanh(pred_verts_incontact_min[ext_and_in_contact] / self.outside_beta) ** 2
            else:
                v2voutside = torch.tensor(0, device=self.device, dtype=self.dtype, requires_grad=True)
            sc_loss = sc_loss + v2vinside.sum() + v2voutside.sum()
        end = time.time()
        print('tuch ld, ', end - start)
        return sc_loss


    def bvh_loss(self, verts):
        pen_loss = torch.tensor(0.0).to(self.device).to(self.dtype)
        # Calculate the loss due to interpenetration
        batch_size = verts.shape[0]
        triangles = verts[:, self.faces[0]]

        with torch.no_grad():
            collision_idxs = self.search_tree(triangles)

        if collision_idxs.ge(0).sum().item() > 0:
            #pen_loss = 1.0 * (torch.tanh(pen_distance(triangles, collision_idxs) / 0.15) ** 2).sum()
            #print(collision_idxs.shape, pen_distance(triangles, collision_idxs).shape)
            #print(collision_idxs[0, :20])
            #print(pen_distance(triangles, collision_idxs)[:20])
            pen_loss = torch.sum(
                    self.pen_distance(triangles, collision_idxs) * 1)
        # if collision_idxs.ge(0).sum().item() > 0:
        #     pen_loss = 100 * torch.exp(100 * pen_distance(triangles, collision_idxs))
        # print(pen_distance(triangles, collision_idxs).max())
        return pen_loss
