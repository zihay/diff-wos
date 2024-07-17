
from dataclasses import dataclass

from numpy import indices
from wos.scene import BoundarySamplingRecord, Detector, Polyline

from wos.solver import Solver
from wos.fwd import *


@dataclass
class WoSBoundary(Solver):
    def walk_detector(self, scene, detector: Detector, seed=0):
        npoints = detector.res[0] * detector.res[1]
        nsamples = npoints * self.nwalks
        sampler = PCG32(size=nsamples, initstate=seed)
        valid, idx, value = self.sample_boundary(scene, detector, sampler)
        result = dr.zeros(type(scene.values), npoints)
        dr.scatter_reduce(dr.ReduceOp.Add, result, value, idx, active=valid)
        size = detector.size()
        return result / size / self.nwalks

    def sample_boundary(self, scene: Polyline, detector: Detector, sampler):
        b_rec: BoundarySamplingRecord = scene.sample_boundary(sampler)
        b_val = self.boundary_term(b_rec) / b_rec.pdf
        valid, idx = detector.index(b_rec.p)
        return valid, idx, b_val

    def boundary_term(self, b_rec: BoundarySamplingRecord):
        xn = dr.dot(b_rec.p, dr.detach(b_rec.n))
        return -dr.detach(b_rec.val) * (xn - dr.detach(xn))
