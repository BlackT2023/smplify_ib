import torch
import numpy as np


def batch_knn(db, queries, geomask=None):
    """
    Batch KNN distance
    :param db: tensor [batch_size, num_points, dimension]
    :param queries: tensor [batch_size, num_queries, dimension]
    :return: nearest points: tensor [batch_size, num_queries, dimension]
    """
    with torch.no_grad():
        queries_expanded = queries.unsqueeze(2)
        db_expanded = db.unsqueeze(1)

        if geomask is None:
            dist = torch.norm(db_expanded - queries_expanded, dim=-1)
        else:
            dist = torch.norm((db_expanded - queries_expanded), dim=-1)
            dist[:, geomask] = float('inf')
        _, indices = torch.min(dist, dim=-1)
    return indices.clone().detach()


def seg2segSDF(db, centers, queries, center_idx, geomask=None, centers2=None, center_idx2=None, output='dist'):
    # torch.autograd.set_detect_anomaly(True)
    device = db.device
    dtype = db.dtype
    batch_size = db.shape[0]

    db_idx = batch_knn(db, queries, geomask)
    nearest_points = db[torch.arange(batch_size).unsqueeze(1), db_idx]

    nearest_center = centers[torch.arange(batch_size).unsqueeze(1),
                             center_idx[db_idx]]

    dist_vec = nearest_points - queries

    center_vec = nearest_center - queries

    v2vinside = (dist_vec * center_vec).sum(-1) < 0
    sign_mart = torch.ones([batch_size, queries.shape[1]], dtype=dtype, device=device, requires_grad=False)
    sign_mart[v2vinside] = -1

    dist = torch.norm(dist_vec, dim=-1) * sign_mart

    if output == 'dist':
        return dist
    elif output == 'volume':
        x = torch.max(dist_vec[:, :, 0], dim=-1)[0] - torch.min(dist_vec[:, :, 0], dim=-1)[0]
        y = torch.max(dist_vec[:, :, 1], dim=-1)[0] - torch.min(dist_vec[:, :, 1], dim=-1)[0]
        z = torch.max(dist_vec[:, :, 2], dim=-1)[0] - torch.min(dist_vec[:, :, 2], dim=-1)[0]
        return dist, x * y * z
    elif output == 'joint_dist':
        if center_idx2 is None or centers2 is None:
            print('output is joint_dist, but center_idx2 not provided')
            return None, None
        joint_dist = torch.norm(nearest_center - centers2[torch.arange(batch_size).unsqueeze(1), center_idx2],
                                dim=-1)
        return dist, joint_dist
    elif output == 'dbg':
        for i in range(batch_size):
            be_isected = torch.ones([batch_size, db.shape[1]], dtype=dtype, device=device, requires_grad=False)
            be_isected[i, db_idx[i, v2vinside[i]]] = -1
        return sign_mart, be_isected


def modify_geodistsmpl(geodistsmpl, segments):
    all_segs = ['head', 'torso_thigh', 'torso_upperarm',
                'left_arm', 'right_arm', 'left_thigh', 'right_thigh']
    for seg in all_segs:
        seg_idx = segments.get_segment_vidx(seg)
        geodistsmpl[np.expand_dims(seg_idx, 1), seg_idx] = 0
        cnt = 1
    return geodistsmpl


def point_line_dist(A, B, C):
    # A, B, C: torch.tensor of shape [batch_size, 3]
    # compute the distance between line AB and point C
    AB = B - A
    AC = C - A

    cross_prod = torch.norm(torch.cross(AB, AC, dim=-1), dim=-1)
    AB_length = torch.norm(AB, dim=-1)

    return cross_prod / AB_length
