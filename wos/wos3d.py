from dataclasses import dataclass

from mitsuba import Scene
from wos.fwd import *
from wos.scene3d import ClosestPointRecord3D, Scene3D
from wos.solver import Solver


@dataclass
class WoS3D(Solver):
    epsilon: float = 1e-3
    nwalks: int = 10
    nsteps: int = 32
    double_sided: bool = False

    def rand_on_sphere(self, sampler):
        u = Array2(sampler.next_float64(),
                   sampler.next_float64())
        z = 1. - 2. * u[0]
        r = dr.sqrt(dr.maximum(0., 1. - z * z))
        theta = 2. * dr.pi * u[1]
        return Array3(r * dr.cos(theta), r * dr.sin(theta), z)

    def rand_on_sphere2(self, sampler):
        '''
        return \cos(\theta) and \phi
        '''
        u = Array2(sampler.next_float64(),
                   sampler.next_float64())
        z = 1. - 2. * u[0]  # [-1, 1]
        phi = 2. * dr.pi * u[1]
        return z, phi

    def single_walk_preliminary(self, _p, scene: Scene3D, sampler):
        p = Array3(_p)
        active = Bool(True)
        i = Int(0)
        p_in_shell = Array3(0.)
        d = dr.abs(scene.sdf(p))
        loop = Loop("single_walk", lambda: (
            i, active, d, p, p_in_shell, sampler))
        while loop(i < self.nsteps):
            in_shell = active & (d < self.epsilon)
            p_in_shell[in_shell] = Array3(p)
            active &= ~in_shell
            p[active] = p + self.rand_on_sphere(sampler) * dr.detach(d)
            d[active] = dr.abs(scene.sdf(p))
            i += 1

        its = scene.closest_point(p_in_shell)
        return its

    def single_walk(self, _p, scene: Scene3D, sampler):
        p = Array3(_p)
        d = scene.sdf(p)
        active = Bool(True)
        if not self.double_sided:
            active = d < 0
        its = self.single_walk_preliminary(p, scene, sampler)
        its.valid &= active
        return scene.dirichlet(its)

    def u(self, p, scene: Scene3D, sampler):
        return self.single_walk(p, scene, sampler)

    def grad(self, _p, scene: Scene3D, sampler):
        # ∇u
        x = Array3(_p)
        T = type(scene.values)
        R = scene.sdf(x)
        active = Bool(True)
        if not self.double_sided:
            active = R < 0
        R = dr.abs(R)
        in_shell = active & (R < self.epsilon)
        active &= ~in_shell
        ret = [T(0.), T(0.), T(0.)]
        dir = self.rand_on_sphere(sampler)
        for i in range(2):
            if i == 1:
                dir = -dir
            # sample a point on the first ball
            y = x + dr.detach(R) * dir
            yx = y - x
            #! eq.13 in https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/paper.pdf
            G = 3. / R * yx / R
            # control variates
            u = self.u(y, scene, sampler) - \
                scene.dirichlet(scene.closest_point(x))
            u = dr.select(active, u, T(0.))
            G = dr.select(active, G, Array3(0.))
            ret[0] += G.x * u
            ret[1] += G.y * u
            ret[2] += G.z * u
        return ret[0] / 2., ret[1] / 2., ret[2] / 2.

    def normal_derivative(self, its: ClosestPointRecord3D, scene: Scene3D, sampler,
                          clamping=1e-1):
        # ! if the point is inside the object, its.n points inward
        n = -Array3(its.n)
        p = Array3(its.p)
        u_ref = scene.dirichlet(its)
        R = scene.largest_inscribed_ball(its.c_object())
        c = p - n * R
        cos_theta, phi = self.rand_on_sphere2(sampler)
        # cos_theta = dr.clamp(cos_theta, -1., 1. - 1e-3)  # ! caution
        cos_theta = dr.clamp(cos_theta, -1., 1. - clamping)
        grad = Float(0.)
        f_dir = n  # forward direction
        up = dr.select(dr.abs(f_dir.z) < 0.9,
                       Array3(0., 0., 1.),
                       Array3(1., 0., 0.))
        # perpendicular direction
        p_dir = dr.normalize(dr.cross(f_dir, up))
        for i in range(2):
            # antithetic azimuthal angle
            if i == 1:
                phi = phi + dr.pi
            # sample a point on the largest ball
            r = dr.sqrt(dr.maximum(0., 1. - cos_theta * cos_theta))
            p = c + R * (cos_theta * f_dir +
                         r * (dr.cos(phi) * p_dir +
                              dr.sin(phi) * dr.cross(f_dir, p_dir)))
            d = dr.abs(scene.sdf(p, active=its.valid))
            # estimate u using WoS
            u = dr.select(d < self.epsilon,
                          scene.dirichlet(
                              scene.closest_point(p, active=its.valid)),
                          self.u(p, scene, sampler))
            # derivative of the off-centered Poisson kernel
            P = -1. / (dr.sqrt(2) * (dr.power(1 - cos_theta, 1.5)))
            # control variates
            grad += P * (u - u_ref) / R
        grad /= 2.
        return grad * n.x, grad * n.y, grad * n.z

    def tangent_derivative(self, its, scene: Scene3D, sampler=None):
        f = dr.gather(Array3i, scene.indices, its.prim_id)
        a = dr.gather(Array3, scene.vertices, f.x)
        b = dr.gather(Array3, scene.vertices, f.y)
        c = dr.gather(Array3, scene.vertices, f.z)
        va = dr.gather(type(scene.values), scene.values, f.x)
        vb = dr.gather(type(scene.values), scene.values, f.y)
        vc = dr.gather(type(scene.values), scene.values, f.z)
        ab = b - a
        ac = c - a
        t = dr.normalize(ab)
        f_n = dr.normalize(dr.cross(ab, ac))  # face normal
        n = dr.normalize(dr.cross(f_n, t))
        vab = (vb - va)
        vac = (vc - va)
        gt = vab / dr.norm(ab)
        gn = (vac - gt * dr.dot(ac, t)) / dr.dot(ac, n)
        return (gt * t.x + gn * n.x,
                gt * t.y + gn * n.y,
                gt * t.z + gn * n.z)


@dataclass
class WoS3DCUDA(WoS3D):
    prevent_fd_artifacts: bool = False

    def __post_init__(self):
        # super().__post_init__()
        from wos_ext import WoS3D as CWoS
        self.cwos = CWoS(nwalks=self.nwalks,
                         nsteps=self.nsteps,
                         epsilon=self.epsilon,
                         double_sided=self.double_sided,
                         prevent_fd_artifacts=self.prevent_fd_artifacts)

    def single_walk_preliminary(self, _p, scene, sampler):
        '''
        uses cuda wos
        '''
        p = Array3(_p)
        its = self.cwos.single_walk(p, scene, sampler)
        return its
