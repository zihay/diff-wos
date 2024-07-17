from dataclasses import dataclass

from mitsuba import Scene
from wos.fwd import *
from wos.scene3d import Scene3D
from wos.solver import Solver
from wos.wos3d import WoS3D, WoS3DCUDA


@dataclass
class WoSGradient3D(Solver):
    epsilon: float = 1e-3  # controls condition to stop recursion
    nwalks: int = 10  # number of samples per point queried
    nsteps: int = 32  # maximum depth of a sampled path
    double_sided: bool = False

    def __post_init__(self):
        self.wos = WoS3DCUDA(nwalks=self.nwalks, nsteps=self.nsteps,
                             epsilon=self.epsilon,
                             double_sided=self.double_sided)


@dataclass
class Baseline3D(Solver):
    def rand_on_sphere(self, sampler):
        u = Array2(sampler.next_float64(),
                   sampler.next_float64())
        z = 1. - 2. * u[0]
        r = dr.sqrt(dr.maximum(0., 1. - z * z))
        theta = 2. * dr.pi * u[1]
        return Array3(r * dr.cos(theta), r * dr.sin(theta), z)

    def weight(self, d):  # 0: stop recursion & use boundary value
        # Traditional hard edge
        # return dr.select(d < self.epsilon, 0.0, 1.0)
        # soft edge
        return dr.minimum(dr.abs(d), self.epsilon) / (self.epsilon)

    def single_walk(self, p, scene: Scene3D, sampler):
        d = scene.sdf(p)  # attached
        active = Bool(True)
        if not self.double_sided:
            active = d < 0
        result = type(scene.values)(0.)
        # throughput. becomes 0 when path ends
        beta = dr.select(active, Float(1.), Float(0.))
        d = dr.abs(d)
        for i in range(self.nsteps):
            w = self.weight(d)
            its = scene.closest_point(p)
            result[d > 0] += beta * (1.0 - w) * scene.dirichlet(its)
            beta *= w
            # uniform sampling
            p = p + d * self.rand_on_sphere(sampler)
            d = dr.abs(scene.sdf(p))

        return dr.select(active, result, type(scene.values)(0.))


@dataclass
class Baseline23D(Solver):
    '''
    this class uses CUDA WoS
    '''
    l: float = 5e-3
    normal_derivative_only: bool = False

    def __post_init__(self):
        self.wos = WoS3DCUDA(nwalks=self.nwalks, nsteps=self.nsteps,
                             epsilon=self.epsilon,
                             double_sided=self.double_sided)

    def single_walk(self, _p, scene: Scene3D, sampler):
        with dr.suspend_grad():
            T = type(scene.values)
            p = Array3(_p)
            d = scene.sdf(p)
            is_inside = d < 0
            if self.double_sided:
                is_inside = Bool(True)
            # ! CUDA WoS: detached
            its = self.wos.single_walk_preliminary(p, scene, sampler)
        # compute velocity
        its = scene.get_point(its)
        p_in_shell = its.p + its.n * self.l
        with dr.suspend_grad():
            # evaluate spatial gradient
            if self.normal_derivative_only:
                grad = self.wos.grad(p_in_shell, scene, sampler)
                grad_n = dr.dot(grad, its.n) * its.n
                grad_t = self.wos.tangent_derivative(its, scene, sampler)
                grad_u = [grad_n[0] + grad_t[0],
                          grad_n[1] + grad_t[1],
                          grad_n[2] + grad_t[2]]
            else:
                grad_u = self.wos.grad(p_in_shell, scene, sampler)

        v = its.p
        v = v - dr.detach(v)
        res = -(grad_u[0] * v[0] + grad_u[1] * v[1] + grad_u[2] * v[2])
        return dr.select(is_inside & its.valid, res, T(0.))


@dataclass
class Ours3D(Solver):
    epsilon2: float = 1e-3  # controls condition to stop recursion
    clamping: float = 1e-1  # controls condition to stop recursion

    def __post_init__(self):
        self.wos = WoS3DCUDA(nwalks=self.nwalks, nsteps=self.nsteps,
                             epsilon=self.epsilon,
                             double_sided=self.double_sided)
        self.wos2 = WoS3DCUDA(nwalks=self.nwalks, nsteps=self.nsteps,
                              epsilon=self.epsilon2,
                              double_sided=self.double_sided)

    def single_walk(self, _p, scene: Scene3D, sampler):
        with dr.suspend_grad():
            p = Array3(_p)
            d = scene.sdf(p)
            is_inside = d < 0
            if self.double_sided:
                is_inside = Bool(True)
            T = type(scene.values)
            its = self.wos.single_walk_preliminary(p, scene, sampler)
        its = scene.get_point(its)
        v = its.p
        v = v - dr.detach(v)
        with dr.suspend_grad():
            grad_n = self.wos2.normal_derivative(
                its, scene, sampler, clamping=self.clamping)
            grad_t = self.wos.tangent_derivative(its, scene, sampler)
            grad_u = [grad_n[0] + grad_t[0],
                      grad_n[1] + grad_t[1],
                      grad_n[2] + grad_t[2]]
        res = -(grad_u[0] * v[0] + grad_u[1] * v[1] + grad_u[2] * v[2])
        return dr.select(is_inside & its.valid, res, T(0.))
