from dataclasses import dataclass, field
from enum import Enum
from wos.fwd import *
from wos.scene import Detector
from wos.io import write_exr
from wos.utils import sample_tea_32


class ControlVarianceType(Enum):
    NoControlVariate = 0
    RunningControlVariate = 1
    BoundaryControlVariate = 2
    BoundaryAndRunningControlVariate = 3


@dataclass
class Solver:
    epsilon: float = 0.005  # controls condition to stop recursion
    nwalks: int = 20  # number of samples per point queried
    nsteps: int = 4   # maximum depth of a sampled path
    double_sided: bool = False
    control_variance: ControlVarianceType = ControlVarianceType.NoControlVariate
    antithetic: bool = False  # ! only used for gradient estimation

    def rand_on_circle(self, sampler):
        rnd = sampler.next_float64()
        angle = rnd * 2 * dr.pi
        return Array2(dr.cos(angle), dr.sin(angle))

    def rand_in_half_circle(self, n, sampler):
        angle = sampler.next_float64() * 2 * dr.pi
        dir = Array2(dr.cos(angle), dr.sin(angle))
        return dr.select(dr.dot(n, dir) > 0., dir, -dir)

    def sample_uniform(self, sampler):
        return Float(sampler.next_float64() * 2 * dr.pi)

    def pdf_uniform(self, theta):
        return 1 / (2 * dr.pi)

    def solve(self, p, scene, sampler):
        result = self.walk_reduce(p, scene, sampler)
        dr.eval(result, sampler)  # NOTE important for speed
        return to_torch(Tensor(result, shape=dr.shape(result)))

    def walk(self, p, scene, seed=0):
        return self.walk_reduce(p, scene, seed)

    def walk_reduce(self, p, scene, seed=0):
        npoints = dr.width(p)
        nsamples = npoints * self.nwalks
        result = dr.zeros(type(scene.values), npoints)
        # multiply the wavefront size by nwalks
        idx = dr.arange(Int, nsamples)
        p = dr.repeat(p, self.nwalks)
        if self.nwalks > 1:
            idx //= self.nwalks
        v0, v1 = sample_tea_32(seed, idx)
        dr.eval(v0, v1)
        sampler = PCG32(size=nsamples, initstate=v0, initseq=v1)
        value = self.single_walk(p, scene, sampler)
        dr.scatter_reduce(dr.ReduceOp.Add, result, value, idx)
        if self.nwalks > 1:
            result /= self.nwalks
        return result

    def walk_detector(self, scene, detector: Detector, seed=0):
        p = detector.make_points()
        npixels = dr.width(p)
        nsamples = npixels * self.nwalks
        idx = dr.arange(Int, nsamples)
        v0, v1 = sample_tea_32(seed, idx)
        dr.eval(v0, v1)
        sampler = PCG32(size=nsamples, initstate=v0, initseq=v1)
        if self.nwalks > 1:
            idx //= self.nwalks
        jitter_p = detector.make_jittered_points(sampler, self.nwalks)
        value = self.single_walk(
            jitter_p, scene, sampler)
        result = dr.zeros(type(scene.values), npixels)
        dr.scatter_reduce(dr.ReduceOp.Add, result, value, idx)
        if self.nwalks > 1:
            result /= self.nwalks
        return result

    def single_walk(self, p, scene, sampler):
        raise NotImplementedError


class _CustomSolver(dr.CustomOp):
    def eval(self, fwd_solver, bwd_solver, p, scene, seed=0, dummy=None):
        '''
        scene might have values that are attached to the AD graph
        '''
        self.fwd_solver = fwd_solver
        self.bwd_solver = bwd_solver
        self.p = p
        self.scene = scene
        self.seed = seed
        # the return value should not be attached to the AD graph
        return dr.detach(fwd_solver.walk(p, scene, seed))

    def forward(self):
        print("forward")
        value = self.fwd_solver.walk(self.p, self.scene, self.seed)
        dr.forward_to(value)
        self.set_grad_out(value)

    def backward(self):
        value = self.bwd_solver.walk(self.p, self.scene, self.seed)
        dr.set_grad(value, self.grad_out())
        dr.enqueue(dr.ADMode.Backward, value)
        dr.traverse(type(value), dr.ADMode.Backward,
                    dr.ADFlag.ClearInterior)  # REVIEW


@dataclass
class CustomSolver():
    fwd_solver: Solver = None
    bwd_solver: Solver = None

    def walk(self, p, scene, seed=0):
        dummy = Float(0.)
        dr.enable_grad(dummy)
        return dr.custom(_CustomSolver, self.fwd_solver, self.bwd_solver, p, scene, seed, dummy)
