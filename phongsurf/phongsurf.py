# Phong Surface
from . import triwalk

class PhongSurface():
    def __init__(self, V, F, N, FN=None):
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

        if self.FN is None:
            if self.F is not None and self.N is not None:
                self.FN = self.F

        self.init_triwalk(self.F)

    def init_triwalk(self, F):
        self.triwalk = None
        if F is not None and F.shape[0] > 0 and F.shape[1] == 3:
            self.triwalk = triwalk.Triwalk(F)

    def __str__(self) -> str:
        return '[PhongSurface] V: %s, F: %s' % (
            str(self.V.shape[0]) if self.V is not None else 'None',
            str(self.F.shape[0]) if self.F is not None else 'None'
        ) + \
            '\n[PhongSurface] N: %s, FN: %s' % (
            str(self.N.shape[0]) if self.N is not None else 'None',
            str(self.FN.shape[0]) if self.FN is not None else 'None'
        )


