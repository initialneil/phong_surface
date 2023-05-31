# Phong Surface
from . import triwalk
import numpy as np
from numpy import linalg as LA

try:
    from scipy import spatial
    import torch
    import torch.nn.functional as F
except:
    print('[PhongSurface] scipy is needed for init corres')
    print('[PhongSurface] torch is needed for solving corres')

# interpolate barycentric attributes
def interpBarycentric(tri_V, spt_vw):
    bary = None
    if type(tri_V) == np.ndarray:
        bary = np.zeros([spt_vw.shape[0], 3], dtype=spt_vw.dtype)
    else:
        bary = torch.zeros([spt_vw.shape[0], 3], dtype=spt_vw.dtype)

    bary[:, 0:2] = spt_vw
    bary[:, 2] = 1.0 - spt_vw[:, 0] - spt_vw[:, 1]
    return (bary[:, None, :] @ tri_V).squeeze(1)

class PhongSurface():
    def __init__(self, V, F, N, FN=None, TC=None, FTC=None, verbose=True):
        """
        PhongSurface based on a triangle mesh.
        - V: vertices
        - F: faces indexing vertices
        - N: vertex normals
        - FN: faces indexing vertex normals
        """
        self.V = V
        self.F = F
        self.N = N
        self.FN = FN
        self.TC = TC
        self.FTC = FTC
        self.verbose = verbose

        if self.FN is None:
            if self.F is not None and self.N is not None:
                self.FN = self.F

        self.init_triwalk(self.F)

        self.source_V = None
        self.source_F = None
        self.source_N = None
        self.source_FN = None

        if self.V is not None:
            self.source_V = torch.from_numpy(self.V).to(dtype=torch.float)
        if self.F is not None:
            self.source_F = torch.from_numpy(self.F).to(dtype=torch.long)
        if self.N is not None:
            self.source_N = torch.from_numpy(self.N).to(dtype=torch.float)
        if self.FN is not None:
            self.source_FN = torch.from_numpy(self.FN).to(dtype=torch.long)

        self.lambda_N = 0.001
        self.outer_loop = 4
        self.inner_loop = 50

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

        spt_fidx, spt_vw = self.triwalk.updateSurfacePoints(spt_fidx, spt_vw, spt_delta)
        return spt_fidx, spt_vw

    # find corresponding surface points for V(:, 3)
    def find_corres_spt(self, V, N=None):
        if V.ndim == 1:
            V = V[None, :]
        if N is not None and N.ndim == 1:
            N = N[None, :]

        # init corresponding surface points
        spt_fidx, spt_vw = self.init_corres_spt(V)

        # solve for update
        spt_fidx, spt_vw = self.update_corres_spt(V, N, spt_fidx, spt_vw)

        return spt_fidx, spt_vw

    # init corresponding surface points
    def init_corres_spt(self, V):
        F_centers = ((self.V[self.F[:, 0], :] + self.V[self.F[:, 1], :] + self.V[self.F[:, 2], :]) / 3)
        tree = spatial.KDTree(F_centers)
        spt_fidx = tree.query(V)[1].astype(np.int32)
        spt_vw = np.ndarray([spt_fidx.shape[0], 2]).astype(np.double)
        spt_vw.fill(1.0 / 3.0)
        return spt_fidx, spt_vw

    # solve for update
    def update_corres_spt(self, V, N, spt_fidx, spt_vw):            
        if V.shape[0] != spt_fidx.shape[0]:
            print('[PhongSurface][ERROR] V and spt_fidx dimension mismatch')
        if V.shape[0] != spt_vw.shape[0]:
            print('[PhongSurface][ERROR] V and spt_vw dimension mismatch')

        # prepare data
        query_V = torch.from_numpy(V).to(dtype=torch.float)
        query_N = torch.from_numpy(N).to(dtype=torch.float) if N is not None else None

        if self.verbose:
            print('[PhongSurface] update_corres_spt ...', end='')

        # outer loop
        alpha = 1.0
        for outer in range(self.outer_loop):
            corres_fidx = torch.from_numpy(spt_fidx).to(dtype=torch.long)
            corres_vw = torch.from_numpy(spt_vw).to(dtype=torch.float)

            # solve for update of barycentric coords
            delta_vw = self.solve_delta_vw(query_V, query_N, corres_fidx, corres_vw)
            spt_delta = delta_vw.numpy().astype(dtype=np.double) * alpha

            # triangle walk for delta        
            spt_fidx, spt_vw = self.triwalk_update(spt_fidx, spt_vw, spt_delta)

            # decay
            alpha = alpha * 0.5

        if self.verbose:
            print('[done]')
        return spt_fidx, spt_vw

    # solve for update of barycentric coords
    def solve_delta_vw(self, query_V, query_N, corres_fidx, corres_vw):
        num = query_V.shape[0]
        x = torch.zeros([num, 2], dtype=torch.float, requires_grad=True)

        # optimizer = torch.optim.SGD([x], lr=1.0)
        # optimizer = torch.optim.Adagrad([x], lr=0.1)
        optimizer = torch.optim.Adagrad([x], lr=0.2, lr_decay=0.1)

        steps = self.inner_loop
        for i in range(steps):
            optimizer.zero_grad()

            if query_N is None:
                corres_V = self.retrieve_vertices(corres_fidx, corres_vw + x)
                loss = torch.sqrt(F.mse_loss(corres_V, query_V))
            else:
                corres_V = self.retrieve_vertices(corres_fidx, corres_vw + x)
                corres_N = self.retrieve_normals(corres_fidx, corres_vw + x)
                # loss = F.l1_loss(corres_V, query_V) + self.lambda_N * F.l1_loss(corres_N, query_N)
                loss = torch.sqrt(F.mse_loss(corres_V, query_V)) + self.lambda_N * torch.sqrt(F.mse_loss(corres_N, query_N))

            loss.backward()
            optimizer.step()
            # print((corres_V[1, :] - query_V[1, :]).norm())
            # if ((corres_V[1, :] - query_V[1, :]).norm() < 6e-5):
            #     print('step =', i)
            #     break

        return x.detach()

    # retrieve vertices for surface points
    def retrieve_vertices(self, spt_fidx, spt_vw):
        if type(spt_fidx) == np.ndarray:
            tri_F = self.F[spt_fidx, :]
            tri_V = np.stack([self.V[tri_F[:, 0], :], self.V[tri_F[:, 1], :], self.V[tri_F[:, 2], :]], axis=1)
            return interpBarycentric(tri_V, spt_vw)
        else:
            tri_F = self.source_F[spt_fidx, :]
            tri_V = torch.stack([self.source_V[tri_F[:, 0], :], self.source_V[tri_F[:, 1], :], self.source_V[tri_F[:, 2], :]], axis=1)
            return interpBarycentric(tri_V, spt_vw)

    # retrieve normals for surface points
    def retrieve_normals(self, spt_fidx, spt_vw):
        if type(spt_fidx) == np.ndarray:
            tri_F = self.FN[spt_fidx, :]
            tri_N = np.stack([self.N[tri_F[:, 0], :], self.N[tri_F[:, 1], :], self.N[tri_F[:, 2], :]], axis=1)
            norms = interpBarycentric(tri_N, spt_vw)
            return (norms / np.linalg.norm(norms, axis=1)[:, None])
        else:
            tri_F = self.source_FN[spt_fidx, :]
            tri_N = torch.stack([self.source_N[tri_F[:, 0], :], self.source_N[tri_F[:, 1], :], self.source_N[tri_F[:, 2], :]], axis=1)
            norms = interpBarycentric(tri_N, spt_vw)
            return F.normalize(norms, p=2, dim=-1)

    # retrieve texture coordinates for surface points
    def retrieve_tc(self, spt_fidx, spt_vw):
        if self.TC is None or self.FTC is None:
            return
            
        tri_F = self.FTC[spt_fidx, :]
        tri_TC = np.stack([self.TC[tri_F[:, 0], :], self.TC[tri_F[:, 1], :], self.TC[tri_F[:, 2], :]], axis=1)
        return interpBarycentric(tri_TC, spt_vw)

    def __str__(self) -> str:
        return '[PhongSurface] V: %s, F: %s' % (
            str(self.V.shape[0]) if self.V is not None else 'None',
            str(self.F.shape[0]) if self.F is not None else 'None'
        ) + \
            '\n[PhongSurface] N: %s, FN: %s' % (
            str(self.N.shape[0]) if self.N is not None else 'None',
            str(self.FN.shape[0]) if self.FN is not None else 'None'
        )


