from dataclasses import dataclass
from wos.fwd import *
from wos.scene import ClosestPointRecord, Polyline
from wos.wos import WoS
import wos_ext


@dataclass
class WoSWithSource(WoS):
    use_IS_for_greens: bool = False
    control_variates: bool = True
    source_epsilon: float = 0.

    def __post_init__(self):
        super().__post_init__()

        N = 10000
        step = 1. / float(N)
        r = dr.linspace(Float, 0., 1.0 - step, N) + Float(0.5 * step)
        G = dr.log(Float(1.) / r) / (2.0 * dr.pi)
        G_pmf = r * G
        self.step = step
        self.G_distr = mi.DiscreteDistribution(G_pmf)

    """
    eval G(x, p) with x = (0, 0) and p is inside the disk
    """

    def eval_G(self, p, R, r_clamp=1e-4):
        r = dr.norm(p)
        return dr.log(R / dr.maximum(r, r_clamp)) / (2.0 * dr.pi)

    """
    eval Poisson kernel of a disk with radius R
    y is inside the disk and s is on the boundary
    """

    def eval_P(self, center, y, s, R):
        y = y - center
        s = s - center
        r = dr.norm(y)
        c = dr.dot(y, s) / dr.norm(s)
        denom = R * R - 2 * R * c + r * r
        P = (R * R - r * r) * dr.rcp(dr.maximum(denom, 1e-5)) / (2.0 * dr.pi * R)
        return P

    def sample_G(self, R, sampler):
        rnd = Float(sampler.next_float32())
        idx, t, pmf = self.G_distr.sample_reuse_pmf(rnd)
        radius = (t + Float(idx)) * self.step
        pdf = pmf / (R * Float(self.step))
        phi = Float(2. * dr.pi * sampler.next_float32())
        pdf = pdf / Float(2. * dr.pi)
        return Array2(radius * R * dr.cos(phi), radius * R * dr.sin(phi)), pdf

    def single_walk_preliminary(self, _p, scene: Polyline, sampler):
        p = Array2(_p)
        y = Array2(_p)
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

            # Sample a point inside the disk.
            if self.use_IS_for_greens:
                dy, pdf = self.sample_G(dr.abs(d), sampler)
                y[active] = dy
                weight = dr.norm(dy)  # Jacobian term from cartesian to polar
            else:
                dy = self.rand_in_disk(sampler)
                pdf = dr.rcp(dr.pi * dr.detach(d * d))
                y[active] = dy * dr.detach(d)
                weight = Float(1.0)

            G = self.eval_G(y, dr.abs(d))
            G = dr.select(active, G, Float(0.))
            G = dr.select(dr.isfinite(G), G, Float(0.))
            pdf = dr.select(active, pdf, Float(0.))
            active &= (pdf > 1e-5)

            y[active] = y + p
            f = dr.select(active, scene.source_function(y), Float(0.))
            result[active] = result + f * G * weight * dr.rcp(pdf)

            p[active] = p + dr.detach(d) * self.rand_on_circle(sampler)
            d[active] = scene.sdf(p)
            i += 1
        its = scene.closest_point(p)
        its.contrib = result
        return its

    def single_walk(self, _p, scene, sampler):
        p = Array2(_p)
        d = scene.sdf(p)
        active = Bool(True)
        if not self.double_sided:
            active = d < 0
        its = self.single_walk_preliminary(_p, scene, sampler)
        its.valid &= active
        return dr.select(its.valid, scene.dirichlet(its) + its.contrib, Float(0.))

    def grad(self, _p, scene, sampler):
        '''
        evaluates the spatial gradient at a point
        '''
        x = Array2(_p)
        T = type(scene.values)
        R = scene.sdf(x)
        active = Bool(True)
        if not self.double_sided:
            active = R < 0
        R = dr.abs(R)
        in_shell = active & (R < self.epsilon)
        active &= ~in_shell

        # boundary term: sample a point on the first ball
        theta = self.sample_uniform(sampler)
        ret = [T(0.), T(0.)]
        # antithetic
        for i in range(2):
            if i == 1:
                theta = -theta
            y = x + dr.detach(R) * Array2(dr.cos(theta), dr.sin(theta))
            yx = y - x
            G = 2. / R * yx / R
            #! control variates
            u = self.u(y, scene, sampler) - \
                scene.dirichlet(scene.closest_point(x))
            u = dr.select(active, u, T(0.))
            G = dr.select(active, G, Array2(0., 0.))
            ret[0] += G.x * u
            ret[1] += G.y * u
        ret[0] /= 2.
        ret[1] /= 2.

        # interior term: sample a point inside the first ball
        y = x + dr.detach(R) * self.rand_in_disk(sampler)
        r = dr.norm(y - x)
        grad_G = (y - x) / (Float(2.) * dr.pi) * \
            (dr.rcp(r * r) - dr.rcp(R * R))
        f = scene.source_function(y) * dr.pi * R * R
        ret[0] += f * grad_G.x
        ret[1] += f * grad_G.y

        return ret[0], ret[1]

    def normal_derivative(self, its: ClosestPointRecord, scene: Polyline, sampler,
                          override_R=None, clamping=1e-1, antithetic=True,
                          control_variates=True, ball_ratio=1., source_epsilon=0.):
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

        R *= ball_ratio
        c = p - n * R

        # boundary term
        #! walk on boundary
        theta = self.sample_uniform(sampler)
        #! prevent large P
        theta = dr.clamp(theta, clamping, 2 * dr.pi - clamping)
        grad = Float(0.)
        #! antithetic sampling
        _n = 1
        if antithetic:
            _n = 2
        for i in range(_n):
            # antithetic angle
            if i == 1:
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
        grad /= _n

        # interior term
        y = c + R * self.rand_in_disk(sampler)
        f = scene.source_function(y)
        P = self.eval_P(c, y, its.p, R)
        # grad = grad - f * P * dr.pi * R * R

        #! control variate
        if control_variates:
            f_s = scene.source_function(its.p)
            contrib_interior = (f - f_s) * P * dr.pi * R * R + f_s * R / 2.0
        else:
            contrib_interior = f * P * dr.pi * R * R
        #! source epsilon
        _d = dr.norm(y - its.p) / R  # relative distance
        # use finite difference to estimate the derivative of f
        _delta = 1e-4
        df = (scene.source_function(its.p + _delta * its.n) -
              scene.source_function(its.p)) / _delta
        contrib_interior = dr.select(_d < source_epsilon,
                                     df / dr.pi, contrib_interior)

        grad = grad - contrib_interior

        return grad * n.x, grad * n.y


@dataclass
class WoSWithSourceCUDA(WoSWithSource):
    prevent_fd_artifacts: bool = False

    def __post_init__(self):
        super().__post_init__()
        from wos_ext import WoS as CWoS
        self.cwos = CWoS(nwalks=self.nwalks,
                         nsteps=self.nsteps,
                         epsilon=self.epsilon,
                         double_sided=self.double_sided,
                         use_IS_for_greens=self.use_IS_for_greens,
                         prevent_fd_artifacts=self.prevent_fd_artifacts)

    def single_walk_preliminary(self, _p, scene, sampler):
        '''
        uses cuda wos
        '''
        p = Array2(_p)
        its = self.cwos.single_walk(p, scene, sampler)
        return its

    def single_walk(self, _p, scene, sampler):
        p = Array2(_p)
        d = scene.sdf(p)
        active = Bool(True)
        if not self.double_sided:
            active = d < 0
        its = self.single_walk_preliminary(_p, scene, sampler)
        its.valid &= active
        return scene.dirichlet(its) + dr.select(its.valid, its.contrib, Float(0.))
