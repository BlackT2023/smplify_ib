import torch


class BedContactLoss():
    def __init__(self,
                 bed_depth=1.66,
                 segments=None,
                 bc_euclthres=0.02,
                 bc_inside_alpha=2,
                 bc_inside_beta=0.25,
                 bc_outside_alpha=0.02,
                 bc_outside_beta=0.02):
        self.segments = segments
        self.bed_depth = bed_depth
        
        self.euclthres = bc_euclthres
        self.inside_alpha = bc_inside_alpha
        self.inside_beta = bc_inside_beta
        self.outside_alpha = bc_outside_alpha
        self.outside_beta = bc_outside_beta
        pass
    
    
    def __call__(self,
                 verts,
                 bc_type='simple'):
        if bc_type == 'simple':
            return self.simple_contact(verts)
        pass
    
    
    def simple_contact(self, verts):
        batch_size = verts.shape[0]
        
        inside_verts = verts[:, :, 2] > 0
        outside_verts = (verts[:, :, 2] > -self.euclthres) & (~inside_verts)
        
        v2binside = (torch.tanh(verts[:, :, 2][inside_verts] / self.inside_beta) ** 2).sum()
        v2boutside = (torch.tanh(-verts[:, :, 2][outside_verts] / self.outside_beta) ** 2).sum()
        
        return v2binside * self.inside_alpha + v2boutside * self.outside_alpha