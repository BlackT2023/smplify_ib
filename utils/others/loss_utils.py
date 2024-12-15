import torch


def gmof(x, sigma):
    """
    Geman-McClure error function
    https://github.com/nkolot/SPIN/blob/master/smplify/losses.py
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def axis2angle(axisAngle):
    # axisAngle: N * 3
    # ret: angle, N * 3
    batch_size = axisAngle.shape[0]
    device, dtype = axisAngle.device, axisAngle.dtype

    # rot matrix, N * 3 * 3
    angle = torch.norm(axisAngle + 1e-8, dim=1, keepdim=True)
    rot_dir = axisAngle / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    # K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    # angle, N * 3
    sy = torch.sqrt(rot_mat[:, 0, 0] ** 2 + rot_mat[:, 1, 0] ** 2)
    singular = sy < 1e-6

    alpha = torch.atan2(rot_mat[:, 1, 0], rot_mat[:, 0, 0])
    alpha[singular] = torch.atan2(-rot_mat[singular, 1, 2], rot_mat[singular, 1, 1])
    beta = torch.atan2(-rot_mat[:, 2, 0], sy)
    gamma = torch.atan2(rot_mat[:, 2, 1], rot_mat[:, 2, 2])
    gamma[singular] = 0

    return torch.stack((alpha, beta, gamma), dim=1)


def polygon_method(polygon, queries):
    """
    :param polygon: tensor [batch_size, num_points, 3], float32
    :param queries: tensor [batch_size, num_queries, 3], float32
    :return: tensor [batch_size, num_points], boolean
    """
    # Polygon Method to judge if inside
    batch_size, num_queries, _ = queries.shape
    num_points = polygon.shape[2]
    device = queries.device

    queries_cp = queries.clone().detach()
    polygon_cp = polygon.clone().detach()
    queries_cp[:, :, -1] = 0
    polygon_cp[:, :, -1] = 0
    standerd_prod_sign = torch.cross(polygon_cp[:, -1].unsqueeze(1).expand(-1, num_queries, -1) - queries_cp,
                                     (polygon_cp[:, -1] - polygon_cp[:, 0]).unsqueeze(1).expand(-1, num_queries, -1),
                                     dim=-1)[:, :, -1]
    # print(standerd_prod_sign)
    inside = torch.ones([batch_size, num_queries], dtype=torch.bool, device=device, requires_grad=False)
    for i in range(num_points - 1):
        prod_sign = torch.cross(polygon_cp[:, i].unsqueeze(1).expand(-1, num_queries, -1) - queries_cp,
                                (polygon_cp[:, i] - polygon_cp[:, i + 1]).unsqueeze(1).expand(-1, num_queries, -1),
                                dim=-1)[:, :, -1]
        inside[(prod_sign * standerd_prod_sign) < 0] = False
    return inside
