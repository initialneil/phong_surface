# Phong Surface
from . import triwalk
import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d import ops
from accelerate import Accelerator

# interpolate barycentric attributes
def _interp_bary(tri_V, spt_vw):
    bary = torch.concat([spt_vw, 1.0 - spt_vw[..., :1] - spt_vw[..., 1:2]], dim=-1)
    if len(bary.shape) == 2:
        return torch.einsum('nij,ni->nj', tri_V, bary)
    else:
        return torch.einsum('bnij,bni->bnj', tri_V, bary)

class PhongSurfacePy3d(torch.nn.Module):
    """
    PhongSurface based on a triangle mesh.
    - V: vertices
    - F: faces indexing vertices
    - N: vertex normals
    """
    def __init__(self, V, F, N, TC=None, FTC=None, 
                 lambda_N=0.01, outer_loop=2, inner_loop=50,
                 max_dist=0.1, verbose=False):
        super().__init__()

        self.register_buffer('V', torch.Tensor(V).to(dtype=torch.float))
        self.register_buffer('F', torch.Tensor(F).to(dtype=torch.long))
        self.register_buffer('N', torch.Tensor(N).to(dtype=torch.float))
        self.register_buffer('TC', torch.Tensor(TC).to(dtype=torch.float))
        self.register_buffer('FTC', torch.Tensor(FTC).to(dtype=torch.long))
        self.verbose = verbose

        self.init_triwalk(F)

        self.lambda_N = lambda_N
        self.outer_loop = outer_loop
        self.inner_loop = inner_loop
        
        self.max_dist = max_dist
        self.F_centers = self.V[self.F].mean(dim=1)

    def __repr__(self) -> str:
        return '[PhongSurfacePy3d] V: %s, F: %s' % (
            str(self.V.shape[0]) if self.V is not None else 'None',
            str(self.F.shape[0]) if self.F is not None else 'None'
        ) + \
            '\n[PhongSurfacePy3d] N: %s' % (
            str(self.N.shape[0]) if self.N is not None else 'None',
        )

    # triangle walk for barycentric delta        
    def init_triwalk(self, F):
        self.triwalk = None
        if F is not None:
            if F.shape[0] == 0:
                print('[PhongSurface] init failed: F is empty')
            elif F.shape[1] != 3:
                print('[PhongSurface] init failed: F.shape[1] must be 3')
            else:
                self.triwalk = triwalk.Triwalk(F)

    def triwalk_update(self, spt_fidx, spt_vw, spt_delta):
        if self.triwalk is None:
            print('[PhongSurface][ERROR] triwalk is None')
            return spt_fidx, spt_vw + spt_delta

        fidx = spt_fidx.detach().cpu().numpy().astype(np.int32)
        vw = spt_vw.detach().cpu().numpy().astype(np.double)
        delta = spt_delta.detach().cpu().numpy().astype(np.double)
        fidx, vw = self.triwalk.updateSurfacePoints(fidx, vw, delta)

        spt_fidx = torch.from_numpy(fidx).to(device=spt_fidx.device, dtype=spt_fidx.dtype)
        spt_vw = torch.from_numpy(vw).to(device=spt_vw.device, dtype=spt_vw.dtype)

        return spt_fidx, spt_vw
    
    """ forward xyz -> uvd, outlier_mask(>max_dist)
    - xyz: [b, n, 3]
    """
    def forward(self, V, N=None):
        return self.find_corres_spt(V, N=N)

    # find corresponding surface points for V(b, n, 3)
    def find_corres_spt(self, V, N=None):
        # init corresponding surface points
        spt_fidx, spt_vw, outlier_mask = self.init_corres_spt(V)
        valid_idxs = ~outlier_mask

        # solve for update
        if N is not None:
            spt_fidx[valid_idxs], spt_vw[valid_idxs] = self.update_corres_spt(V[valid_idxs], N[valid_idxs], spt_fidx[valid_idxs], spt_vw[valid_idxs])
        else:
            spt_fidx[valid_idxs], spt_vw[valid_idxs] = self.update_corres_spt(V[valid_idxs], None, spt_fidx[valid_idxs], spt_vw[valid_idxs])

        return spt_fidx, spt_vw, outlier_mask

    # init corresponding surface points
    def init_corres_spt(self, x):
        distance_batch, index_batch, neighbor_points = ops.knn_points(x, self.F_centers.unsqueeze(0),
                                                                      K=1, return_nn=True)
        

        outlier_mask = (distance_batch[..., 0] > self.max_dist)
        valid_idxs = ~outlier_mask
    
        spt_fidx = index_batch[..., 0]
        spt_vw = torch.zeros_like(x[..., :2])
        spt_vw.fill_(1.0 / 3.0)
        return spt_fidx, spt_vw, outlier_mask

    # solve for update
    # - V(bn, 3)
    def update_corres_spt(self, V, N, spt_fidx, spt_vw):            
        if self.verbose:
            print('[PhongSurface] update_corres_spt ...', end='')

        # outer loop
        alpha = 1.0
        for outer in range(self.outer_loop):
            # solve for update of barycentric coords
            with torch.enable_grad():
                delta_vw = self.solve_delta_vw(V, N, spt_fidx, spt_vw)
            spt_delta = delta_vw * alpha

            # triangle walk for delta        
            spt_fidx, spt_vw = self.triwalk_update(spt_fidx, spt_vw, spt_delta)

            # decay
            alpha = alpha * 0.5

        if self.verbose:
            print('[done]')
        return spt_fidx, spt_vw

    # solve for update of barycentric coords
    def solve_delta_vw(self, query_V, query_N, corres_fidx, corres_vw):
        accelerator = Accelerator()
        device = accelerator.device

        delta = torch.full(corres_vw.shape, 0.0, device=device, requires_grad=True)

        # optimizer = torch.optim.SGD([delta], lr=1.0)
        # optimizer = torch.optim.Adagrad([delta], lr=0.1)
        optimizer = torch.optim.Adagrad([delta], lr=0.2, lr_decay=0.1)
        optimizer = accelerator.prepare(optimizer)

        steps = self.inner_loop
        for i in range(steps):
            optimizer.zero_grad()

            corres_V = self.retrieve_vertices(corres_fidx, corres_vw + delta)
            loss_v = (corres_V - query_V).norm(dim=-1).mean()

            if query_N is None:
                corres_N = self.retrieve_normals(corres_fidx, corres_vw + delta)
                # loss = torch.sqrt(F.mse_loss(corres_V, query_V))
                connect_N = F.normalize(query_V - corres_V, p=2, dim=-1)
                loss_n = torch.min((connect_N - corres_N).norm(dim=-1), (-connect_N - corres_N).norm(dim=-1))
                loss_n = loss_n.mean()
            else:
                corres_N = self.retrieve_normals(corres_fidx, corres_vw + delta)
                # loss = F.l1_loss(corres_V, query_V) + self.lambda_N * F.l1_loss(corres_N, query_N)
                loss_n = (corres_N - query_N).norm(dim=-1)
                loss_n = loss_n.mean()

            loss = loss_v + self.lambda_N * loss_n

            accelerator.backward(loss)
            optimizer.step()

        return delta.detach()

    # retrieve vertices for surface points
    def retrieve_vertices(self, spt_fidx, spt_vw):
        tri_F = self.F[spt_fidx]
        tri_V = self.V[tri_F]
        return _interp_bary(tri_V, spt_vw)

    # retrieve normals for surface points
    def retrieve_normals(self, spt_fidx, spt_vw):
        tri_F = self.F[spt_fidx]
        tri_N = self.N[tri_F]
        norms = _interp_bary(tri_N, spt_vw)
        return F.normalize(norms, p=2, dim=-1)

    # retrieve texture coordinates for surface points
    def retrieve_tc(self, spt_fidx, spt_vw):
        if self.TC is None or self.FTC is None:
            return None
            
        tri_F = self.FTC[spt_fidx]
        tri_TC = self.TC[tri_F]
        return _interp_bary(tri_TC, spt_vw)
