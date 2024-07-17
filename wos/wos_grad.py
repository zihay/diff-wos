from dataclasses import dataclass

from wos.fwd import *
from wos.scene import Intersection, Polyline
from wos.solver import Solver
from wos.wos import WoS, WoSCUDA

'''
This file is for computing the gradient with respect to
scene parameters. The integrator will return a variable
with zero value but non-zero gradient.
'''


@dataclass
class Baseline2(Solver):
    epsilon: float = 1e-3  # controls condition to stop recursion
    nwalks: int = 10  # number of samples per point queried
    nsteps: int = 32  # maximum depth of a sampled path
    variance_reduction: bool = False
    l: float = 5e-3
    normal_derivative_only: bool = False

    def __post_init__(self):
        self.wos = WoSCUDA(nsteps=self.nsteps, nwalks=self.nwalks,
                           epsilon=self.epsilon,
                           double_sided=self.double_sided)

    def single_walk(self, _p, scene: Polyline, sampler):
        T = type(scene.values)
        with dr.suspend_grad():
            p = Array2(_p)
            d = scene.sdf(p)
            is_inside = Bool(True)
            if not self.double_sided:
                is_inside = d < 0
        with dr.suspend_grad():
            its = self.wos.single_walk_preliminary(p, scene, sampler)
        # nudge the point
        # its = scene.closest_point(p_in_shell)
        its = scene.get_point(its)
        p_in_shell = its.p + its.n * self.l
        with dr.suspend_grad():
            if self.normal_derivative_only:
                grad = self.wos.grad(p_in_shell, scene, sampler)
                grad_n = dr.dot(grad, its.n) * its.n
                grad_t = self.wos.tangent_derivative(its, scene, sampler)
                grad_u = [grad_n[0] + grad_t[0], grad_n[1] + grad_t[1]]
            else:
                grad_u = self.wos.grad(p_in_shell, scene, sampler)
        v = its.p
        v = v - dr.detach(v)
        res = -(grad_u[0] * v[0] + grad_u[1] * v[1])
        return dr.select(is_inside & its.valid, res, T(0.))


@dataclass
class Ours(Solver):
    '''
    uses new normal derivative estimator
    '''

    def __post_init__(self):
        self.wos = WoS(nsteps=self.nsteps, nwalks=self.nwalks,
                       epsilon=self.epsilon,
                       double_sided=self.double_sided)

    def single_walk(self, _p, scene: Polyline, sampler):
        T = type(scene.values)
        with dr.suspend_grad():
            p = Array2(_p)
            d = scene.sdf(p)
            is_inside = Bool(True)
            if not self.double_sided:
                is_inside = d < 0
            its = self.wos.single_walk_preliminary(_p, scene, sampler)

        its = scene.get_point(its)
        v = its.p
        v = v - dr.detach(v)
        with dr.suspend_grad():
            grad_n = self.wos.normal_derivative(its, scene, sampler)
            grad_t = self.wos.tangent_derivative(its, scene, sampler)
            grad_u = [grad_n[0] + grad_t[0], grad_n[1] + grad_t[1]]
        res = -(grad_u[0] * v[0] + grad_u[1] * v[1])
        return dr.select(is_inside & its.valid, res, T(0.))


@dataclass
class Baseline2CUDA(Solver):
    '''
    this class uses CUDA WoS
    '''

    def single_walk(self, _p, scene: Polyline, sampler):
        T = type(scene.values)
        p = Array2(_p)
        # ! CUDA WoS: detached
        its = self.wos.cwos.single_walk(p, scene, sampler)
        # compute velocity
        its = scene.get_point(its)
        # ! its.n is the outward normal
        p_in_shell = its.p - its.n * 5e-3
        with dr.suspend_grad():
            # evaluate spatial gradient
            grad_u = self.wos.grad(p_in_shell, scene, sampler)
        v = its.p
        v = v - dr.detach(v)
        res = -(grad_u[0] * v[0] + grad_u[1] * v[1])
        return dr.select(its.valid, res, T(0.))


@dataclass
class OursCUDA(Solver):
    '''
    uses new normal derivative estimator + CUDA WoS
    '''
    epsilon2: float = 1e-3
    clamping: float = 1e-1
    control_variates: bool = True

    def __post_init__(self):
        self.wos = WoSCUDA(nsteps=self.nsteps, nwalks=self.nwalks,
                           epsilon=self.epsilon,
                           double_sided=self.double_sided)
        self.wos2 = WoSCUDA(nsteps=self.nsteps, nwalks=self.nwalks,
                            epsilon=self.epsilon2,
                            double_sided=self.double_sided)

    def single_walk(self, _p, scene: Polyline, sampler):
        p = Array2(_p)
        T = type(scene.values)
        with dr.suspend_grad():
            p = Array2(_p)
            d = scene.sdf(p)
            is_inside = Bool(True)
            if not self.double_sided:
                is_inside = d < 0
        # ! CUDA WoS: detached
        with dr.suspend_grad():
            its = self.wos.single_walk_preliminary(p, scene, sampler)
        its = scene.get_point(its)
        v = its.p
        v = v - dr.detach(v)
        with dr.suspend_grad():
            grad_n = self.wos2.normal_derivative(its, scene, sampler, clamping=self.clamping,
                                                 control_variates=self.control_variates)
            grad_t = scene.tangent_derivative(its)
            grad_u = [grad_n[0] + grad_t[0], grad_n[1] + grad_t[1]]
        res = -(grad_u[0] * v[0] + grad_u[1] * v[1])
        return dr.select(is_inside & its.valid, res, T(0.))
