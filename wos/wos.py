from dataclasses import dataclass
from enum import Enum
from random import randint
from wos.fwd import *
from wos.scene import ClosestPointRecord, Intersection, Polyline
from wos.solver import ControlVarianceType, Solver
import wos_ext


@dataclass
class WoS(Solver):
    epsilon: float = 1e-3  # controls condition to stop recursion
    nwalks: int = 10  # number of samples per point queried
    nsteps: int = 32  # maximum depth of a sampled path
    double_sided: bool = False

    def __post_init__(self):
        if self.control_variance == ControlVarianceType.RunningControlVariate:
            assert self.is_loop

    def rand_in_disk(self, sampler):
        r = dr.sqrt(sampler.next_float64())
        angle = sampler.next_float64() * 2 * dr.pi
        return Array2(r * dr.cos(angle), r * dr.sin(angle))

    def single_walk_preliminary(self, _p, scene: Polyline, sampler):
        p = Array2(_p)
        T = type(scene.values)
        d = scene.sdf(p)
        result = T(0.)
        active = Bool(True)
        i = Int(0)
        p_in_shell = Array2(0.)
        loop = Loop("single_walk", lambda: (i, active, d, p, result, sampler))
        while loop(i < self.nsteps):
            in_shell = active & (dr.abs(d) < self.epsilon)
            p_in_shell[in_shell] = p
            active &= ~in_shell

            p[active] = p + dr.detach(d) * self.rand_on_circle(sampler)
            d[active] = scene.sdf(p)
            i += 1
        its = scene.closest_point(p)
        return its

    def single_walk(self, _p, scene: Polyline, sampler):
        p = Array2(_p)
        d = scene.sdf(p)
        active = Bool(True)
        if not self.double_sided:
            active = d < 0
        its = self.single_walk_preliminary(_p, scene, sampler)
        its.valid &= active
        return dr.select(its.valid, scene.dirichlet(its), type(scene.values)(0.))

    def u(self, _p, scene, sampler):
        return self.single_walk(_p, scene, sampler)

    def normal_derivative(self, its: ClosestPointRecord, scene: Polyline, sampler,
                          override_R=None, clamping=1e-1, control_variates=True):
        # ! if the point is inside the object, its.n points inward
        n = -Array2(its.n)
        p = Array2(its.p)
        u_ref = scene.dirichlet(its)
        # find the largest ball
        # i = Int(0)
        # loop = Loop("normal_derivative", lambda: (i, p, n))
        # while loop(i < 10):
        #! assume a square geometry
        # R = 0.5 - dr.minimum(dr.abs(p.x), dr.abs(p.y))
        if override_R is None:
            R = scene.largest_inscribed_ball(its.c_object())
        else:
            R = override_R
        c = p - n * R
        #! walk on boundary
        theta = self.sample_uniform(sampler)
        #! prevent large P
        theta = dr.clamp(theta, clamping, 2 * dr.pi - clamping)
        grad = Float(0.)
        #! antithetic sampling
        # state = sampler.state + 0
        for i in range(2):
            # antithetic angle
            if i == 1:
                # sampler.state = state
                theta = -theta
            # forward direction
            f_dir = n
            # perpendicular direction
            p_dir = Array2(-f_dir.y, f_dir.x)
            # sample a point on the largest ball
            p = c + R * Array2(f_dir * dr.cos(theta) + p_dir * dr.sin(theta))
            d = dr.abs(scene.sdf(p))
            # start wos to estimate u
            u = dr.select(d < self.epsilon,
                          scene.dirichlet(scene.closest_point(p)),
                          self.single_walk(p, scene, sampler))
            # derivative of off-centered Poisson kernel
            P = 1. / (dr.cos(theta) - 1.)
            # control variate
            if control_variates:
                grad += P * (u - u_ref) / R
            else:
                grad += P * u / R
        grad /= 2.
        return grad * n.x, grad * n.y

    def tangent_derivative(self, its, scene: Polyline, sampler):
        # with dr.resume_grad():
        #     # dudt
        #     t = dr.detach(Float(its.t))
        #     dr.enable_grad(t)
        #     dr.set_grad(t, Float(1.))
        #     u = scene.dirichlet(ClosestPointRecord(
        #         valid=True, prim_id=its.prim_id, t=t))
        #     dr.forward_to(u)
        #     dudt = dr.grad(u)
        #     # dt/dx dt/dy
        #     # t_d = Array2(-dr.sin(t), dr.cos(t)) #! For Circle: scene.radius
        f = dr.gather(Array2i, scene.indices, its.prim_id)
        v0 = dr.gather(Array2, scene.vertices, f.x)
        v1 = dr.gather(Array2, scene.vertices, f.y)
        val0 = dr.gather(Float, scene.values, f.x)
        val1 = dr.gather(Float, scene.values, f.y)
        dv = (val1 - val0) / dr.norm(v1 - v0)
        d_t = dr.normalize(v1 - v0)
        # d_t = Array2(its.n.y, -its.n.x)  # tangent direction
        return dv * d_t.x, dv * d_t.y

        # l = dr.normalize(v1 - v0)
        # t_d = Array2(its.n.y, -its.n.x)
        # return dudt * t_d.x, dudt * t_d.y


@dataclass
class WoSCUDA(WoS):
    '''
    uses cuda implementation
    '''
    prevent_fd_artifacts: bool = False

    def __post_init__(self):
        super().__post_init__()
        from wos_ext import WoS as CWoS
        self.cwos = CWoS(nwalks=self.nwalks,
                         nsteps=self.nsteps,
                         epsilon=self.epsilon,
                         double_sided=self.double_sided,
                         prevent_fd_artifacts=self.prevent_fd_artifacts)

    def single_walk_preliminary(self, _p, scene, sampler):
        p = Array2(_p)
        its = self.cwos.single_walk(p, scene, sampler)
        return its


@dataclass
class Baseline(Solver):
    epsilon: float = 0.005  # controls condition to stop recursion
    nwalks: int = 10  # number of samples per point queried
    nsteps: int = 32  # maximum depth of a sampled path

    def weight(self, d):  # 0: stop recursion & use boundary value
        # Traditional hard edge
        # return dr.select(d < self.epsilon, 0.0, 1.0)
        # soft edge
        return dr.minimum(dr.abs(d), self.epsilon) / (self.epsilon)

    def single_walk(self, p, scene: Polyline, sampler):
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
            p = p + d * self.rand_on_circle(sampler)
            d = dr.abs(scene.sdf(p))
        return dr.select(active, result, type(scene.values)(0.))
