from dataclasses import dataclass
from typing import List
from wos.fwd import *
from wos.io import write_exr
import numpy as np
import wos_ext


@dataclass
class Detector:
    vmin: tuple = (-1., -1.)
    vmax: tuple = (1., 1.)
    res: tuple = (100, 100)

    def size(self):
        return dr.prod(Array2(self.vmax) - Array2(self.vmin))

    def index(self, p):
        uv = (p - Array2(self.vmin)) / (Array2(self.vmax) - Array2(self.vmin))
        uv = uv * Array2(self.res[0], self.res[1]) + Array2(0.5, 0.5)
        uv = Array2i(uv)
        valid = (uv.x >= 0) & (uv.x < self.res[0]) & \
                (uv.y >= 0) & (uv.y < self.res[1])
        return valid, uv.x + (self.res[1] - uv.y) * self.res[0]

    def make_points(self):
        x = dr.linspace(Float, self.vmin[0], self.vmax[0], self.res[0])
        y = dr.linspace(Float, self.vmax[1],
                        self.vmin[1], self.res[1])  # flip y
        p = Array2(dr.meshgrid(x, y))
        return p

    def make_jittered_points(self, sampler, spp):
        p = self.make_points()
        p = dr.repeat(p, spp)
        d = (Array2(self.vmax) - Array2(self.vmin)) / Array2(self.res)
        p += Array2(sampler.next_float64() - 0.5,
                    sampler.next_float64() - 0.5) * d
        return p

    def save(self, image, filename):
        write_exr(filename, image)


@dataclass
class Intersection:
    valid: Bool = Bool()
    p: Array2 = Array2()
    n: Array2 = Array2()
    val: Float = Float()
    prim_id: Int = Int(-1)
    t_seg: Float = Float(0.)
    t: Float = Float(0.)
    count: Int = Int(1)  # number of intersections
    J: Float = Float(1.)  # Jacobian

    DRJIT_STRUCT = {'valid': Bool, 'p': Array2, 'n': Array2, 'val': Float,
                    'prim_id': Int, 't_seg': Float, 't': Float, 'count': Int, 'J': Float}

    def is_inside(self):
        # use winding number to determine if the ray origin is inside the shape
        return dr.eq(self.count % 2, 1)  # NOTE: can't use self.count % 2 == 1


@dataclass
class ClosestPointRecord:
    valid: Bool = Bool()
    p: Array2 = Array2()
    n: Array2 = Array2()
    t: Float = Float()
    val: Float = Float()
    prim_id: Int = Int(-1)
    J: Float = Float(1.)
    contrib: Float = Float(0.)

    def c_object(self):
        return wos_ext.ClosestPointRecord(self.valid,
                                          self.p,
                                          self.n,
                                          self.t,
                                          self.prim_id,
                                          self.contrib)


@dataclass
class BoundarySamplingRecord:
    p: Array2 = Array2()  # ! attached(get_point)
    n: Array2 = Array2()  # ! detached
    val: Float = Float()  # ! onesided Dirichlet
    prim_id: Int = Int(-1)
    t: Float = Float()
    pdf: Float = Float(1.)


@dataclass
class Scene:
    def sdf(self, p: Array2) -> Float:
        raise NotImplementedError

    def dirichlet(self, p: Array2) -> Float:
        raise NotImplementedError

    def source_function(self, p: Array2) -> Float:
        raise NotImplementedError


@dataclass
class Segment:
    a: Array2 = Array2(Float(0.), Float(0.))
    b: Array2 = Array2(Float(1.), Float(0.))
    va: Float = 0.
    vb: Float = 1.
    DRJIT_STRUCT = {'a': Array2, 'b': Array2,  'va': Float, 'vb': Float}

    def sdf(self, p: Array2) -> Float:
        pa = p - self.a
        ba = self.b - self.a
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0, 1)
        d = dr.norm(pa - ba * h)
        return d

    def dirichlet(self, p: Array2) -> Float:
        pa = p - self.a
        ba = self.b - self.a
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0, 1)
        d = dr.norm(pa - ba * h)
        # assert dr.all(dr.abs(d) < 1e-2)  # make sure p is near the segment
        s = pa[0] * ba[1] - pa[1] * ba[0]
        v = dr.lerp(self.va, self.vb, h)
        return dr.select(s < 0, v, 0.)


@dataclass
class Circle(Scene):
    rotation: Float = Float(0.)
    center: Array2 = Array2(0.)
    radius: Float = Float(1.)

    def _sdf(self, p: Array2) -> Float:
        return dr.norm(p - self.center) - self.radius

    def sdf(self, p: Array2) -> Float:
        return self._sdf(p)

    def sdf_grad(self, p: Array2) -> Array2:
        return dr.normalize(p - self.center)

    def closest_point(self, p: Array2):
        theta = dr.atan2(p.y - self.center.y, p.x -
                         self.center.x) - self.rotation  # ! careful
        closest_point = dr.normalize(
            p - self.center) * self.radius + self.center
        n = dr.normalize(p - self.center)
        n *= dr.sign(dr.dot(n, p - closest_point))
        return ClosestPointRecord(valid=Bool(True),
                                  p=closest_point,
                                  n=n,
                                  t=theta,
                                  val=None,
                                  prim_id=Int(-1))

    def dirichlet(self, its: ClosestPointRecord) -> Float:
        its.val = its.t  # side effect : inplace modification
        return its.val

    def largest_inscribed_ball(self, its: ClosestPointRecord):
        return self.radius * 0.9

    def analytic_solve(self, p: Array2) -> Float:
        d = dr.norm(p) - self.radius
        active = d < 0
        return dr.select(active, (p.x - self.center.x) / self.radius, 0.)

    def get_point(self, its: ClosestPointRecord):
        active = its.valid
        t = dr.detach(its.t)
        p = self.center + self.radius * \
            Array2(dr.cos(t+self.rotation),
                   dr.sin(t+self.rotation))  # ! careful
        v = self.dirichlet(its)
        J = self.radius / dr.detach(self.radius)
        return ClosestPointRecord(valid=active,
                                  p=p,
                                  n=its.n,
                                  t=t,
                                  val=v,
                                  prim_id=its.prim_id,
                                  J=J)


# @dataclass
class Polyline(wos_ext.Scene):
    vertices: Array2  # use c_scene
    indices: Array2i  # use c_scene
    values: Float  # = None
    # normals: Array2 # use c_scene
    use_bvh: bool = False

    edge_distr: mi.DiscreteDistribution = None

    # Parameters of an analytical source function f.
    source_type: int = 0
    source_params: Array2 = None

    @staticmethod
    def from_vertices(vertices: Array2, values: Float):
        indices = np.array([[i, (i + 1) % dr.width(vertices)]
                            for i in range(dr.width(vertices))])
        return Polyline(vertices, Array2i(indices), values)

    @staticmethod
    def from_obj(filename, normalize=True, flip_orientation=False):
        # read obj
        vertices = []
        indices = []
        values = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    v = [float(x) for x in line[2:].split(' ')]
                    vertices.append(v[:2])
                    # values.append(v[1])
                elif line.startswith('l '):
                    l = [int(x) - 1 for x in line[2:].split(' ')]
                    if flip_orientation:
                        l = l[::-1]
                    indices.append(l)
                elif line.startswith('c '):
                    c = [float(x) for x in line[2:].split(' ')]
                    values.append(c[0])
        if normalize:
            cm = np.mean(vertices, axis=0)
            vertices = [v - cm for v in vertices]
            r = max([np.linalg.norm(v) for v in vertices])
            vertices = [v / r for v in vertices]
        vertices = Array2(np.array(vertices))
        indices = Array2i(np.array(indices))
        values = Float(np.array(values))
        return Polyline(vertices, indices, values)

    def __init__(self, vertices, indices, values, source_type=0, source_params=None, use_bvh=False):
        #! build the c_scene
        # self.values = values
        # if len(dr.shape(values)) > 1:
        #     print('warning: values should be a scalar')
        #     values = values.x
        if source_type == 0:
            super().__init__(vertices, indices, values, use_bvh=use_bvh)
        else:
            assert source_params is not None
            super().__init__(vertices, indices, values, use_bvh, source_type, source_params)
        # self.vertices = vertices
        # self.indices = indices
        # self.values = values
        # normals are attached to the graph
        # p0 = dr.gather(Array2, self.vertices, self.indices.x)
        # p1 = dr.gather(Array2, self.vertices, self.indices.y)
        # d = p1 - p0
        # n = dr.normalize(Array2(d.y, -d.x))
        # self.face_normals = dr.detach(n)
        # self.normals = dr.zeros(Array2, shape=dr.width(vertices))
        # dr.scatter_reduce(dr.ReduceOp.Add, self.normals, n, self.indices.x)
        # dr.scatter_reduce(dr.ReduceOp.Add, self.normals, n, self.indices.y)

        # self.normals = dr.normalize(self.normals)
        # self.normals = dr.detach(self.normals)

        # init edge distribution for edge sampling
        self.init_dist(vertices, indices)

        self.source_type = source_type
        self.source_params = source_params

    def init_dist(self, vertices, indices):
        v0 = dr.gather(type(self.vertices), vertices, indices.x)
        v1 = dr.gather(type(self.vertices), vertices, indices.y)
        l = dr.norm(v1 - v0)
        self.edge_distr = mi.DiscreteDistribution(dr.detach(l))

    def closest_point_preliminary(self, p: Array2):
        '''
        this function should be detached from AD graph
        '''
        with dr.suspend_grad():
            d = Float(dr.inf)
            idx = Int(-1)
            i = Int(0)
            loop = Loop("closest_point", lambda: (idx, d, i))
            while loop(i < dr.width(self.indices)):
                f = dr.gather(Array2i, self.indices, i)
                a = dr.gather(Array2, self.vertices, f.x)
                b = dr.gather(Array2, self.vertices, f.y)
                pa = p - a
                ba = b - a
                h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0, 1)
                _d = dr.norm(pa - ba * h)  # distance to the current primitive
                idx = dr.select(_d < d, i, idx)
                d = dr.select(_d < d, _d, d)
                i += 1
        return idx

    def closest_point(self, p):
        idx = self.closest_point_preliminary(p)
        f = dr.gather(Array2i, self.indices, idx)
        a = dr.gather(Array2, self.vertices, f.x)
        b = dr.gather(Array2, self.vertices, f.y)
        va = dr.gather(type(self.values), self.values, f.x)
        vb = dr.gather(type(self.values), self.values, f.y)

        pa = p - a
        ba = b - a
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0, 1)
        d = dr.norm(pa - ba * h)
        valid = (idx >= 0)
        n = dr.normalize(Array2(ba.y, -ba.x))
        n *= dr.sign(dr.dot(n, pa))
        return ClosestPointRecord(valid=valid,
                                  p=dr.lerp(a, b, h),
                                  n=n,
                                  t=h,
                                  val=dr.lerp(va, vb, h),
                                  prim_id=idx)

    def sdf(self, p: Array2) -> Float:
        idx = self.closest_point_preliminary(p)
        f = dr.gather(Array2i, self.indices, idx)
        a = dr.gather(Array2, self.vertices, f.x)
        b = dr.gather(Array2, self.vertices, f.y)
        pa = p - a
        ba = b - a
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0, 1)
        d = pa - ba * h

        na = dr.gather(Array2, self.normals, f.x)
        nb = dr.gather(Array2, self.normals, f.y)
        n = dr.lerp(na, nb, h)

        return dr.sign(dr.dot(n, d)) * dr.norm(d)

    def sdf_grad(self, p: Array2) -> Float:
        g = Array2(0., 0.)  # projected
        idx = self.closest_point_preliminary(p)
        f = dr.gather(Array2i, self.indices, idx)
        a = dr.gather(Array2, self.vertices, f.x)
        b = dr.gather(Array2, self.vertices, f.y)
        pa = p - a
        ba = b - a
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0, 1)
        g = a + ba * h
        return dr.normalize(g - p)

    def dirichlet(self, its: ClosestPointRecord):
        f = dr.gather(Array2i, self.indices, its.prim_id)
        va = dr.gather(type(self.values), self.values, f.x)
        vb = dr.gather(type(self.values), self.values, f.y)
        return dr.select(its.valid, dr.lerp(va, vb, its.t), 0.)

    def source_function(self, p):
        """
        source_type: 
            0 - none.
            1 - Gaussian: params = [mu (2 floats), sigma (2 floats), amplitude (1 float)]
            2 - sinusoid: params = [A (1 float), B (1 float), C (1 float), amplitude (1 float), offset (1 float)]
        """

        if self.source_type == 0:
            f = Float(0.)
        elif self.source_type == 1:
            dx = (p.x - self.source_params[0]) * (p.x - self.source_params[0]) / (
                Float(2.0) * self.source_params[2] * self.source_params[2])
            dy = (p.y - self.source_params[1]) * (p.y - self.source_params[1]) / (
                Float(2.0) * self.source_params[3] * self.source_params[3])
            f = self.source_params[4] * dr.exp(-dx - dy)
        elif self.source_type == 2:
            phi = p.x * self.source_params[0] + p.y * \
                self.source_params[1] + self.source_params[2]
            f = self.source_params[3] * dr.sin(phi) + self.source_params[4]
        else:
            f = Float(0.)
        return f

    def distance(self, p, prim_id, active):
        f = dr.gather(Array2i, self.indices, prim_id, active)
        a = dr.gather(Array2, self.vertices, f.x, active)
        b = dr.gather(Array2, self.vertices, f.y, active)
        pa = p - a
        ba = b - a
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0, 1)
        d = dr.norm(pa - ba * h)
        return dr.abs(d)

    def boundary_angle(self, p, prim_id, active):
        f = dr.gather(Array2i, self.indices, prim_id, active)
        p0 = dr.gather(Array2, self.vertices, f.x, active)
        p1 = dr.gather(Array2, self.vertices, f.y, active)
        d0 = dr.normalize(p0 - p)
        d1 = dr.normalize(p1 - p)
        angle = dr.acos(dr.dot(d0, d1))
        return angle

    def geometric(self, x: Array2, y: Array2, n: Array2):
        d = y - x
        r = dr.norm(d)
        d = d / r
        cos = dr.dot(d, n)
        return cos / (2 * dr.pi * r)

    def J(self, prim_id):
        f = dr.gather(Array2i, self.indices, prim_id)
        a = dr.gather(Array2, self.vertices, f.x)
        b = dr.gather(Array2, self.vertices, f.y)
        l = dr.norm(b - a)
        return l / dr.detach(l)

    def get_point(self, its: ClosestPointRecord):
        active = its.valid
        t = dr.detach(its.t)
        f = dr.gather(Array2i, self.indices, its.prim_id, active=active)
        p0 = dr.gather(Array2, self.vertices, f.x, active=active)
        p1 = dr.gather(Array2, self.vertices, f.y, active=active)
        v0 = dr.gather(type(self.values), self.values, f.x, active=active)
        v1 = dr.gather(type(self.values), self.values, f.y, active=active)
        p = dr.lerp(p0, p1, t)
        v = dr.lerp(v0, v1, t)
        J = self.J(its.prim_id)
        return ClosestPointRecord(valid=active,
                                  p=p,
                                  n=its.n,
                                  t=t,
                                  val=v,
                                  prim_id=its.prim_id,
                                  J=J)  # FIXME t=its.t

    def sample_boundary(self, sampler):
        rnd = Float(sampler.next_float32())
        prim_id, t, prob = self.edge_distr.sample_reuse_pmf(rnd)
        f = dr.gather(Array2i, self.indices, prim_id)
        a = dr.gather(Array2, self.vertices, f.x)
        b = dr.gather(Array2, self.vertices, f.y)
        ab = b - a
        va = dr.gather(Float, self.values, f.x)
        vb = dr.gather(Float, self.values, f.y)
        val = dr.lerp(va, vb, t)  # ! one-sided Dirichlet
        x = dr.lerp(a, b, t)
        n = dr.detach(dr.normalize(Array2(-ab.y, ab.x)))
        return BoundarySamplingRecord(p=x, n=n, val=val,
                                      prim_id=prim_id, t=rnd,
                                      pdf=prob / dr.norm(ab))

    def tangent_derivative(self, its: ClosestPointRecord):
        f = dr.gather(Array2i, self.indices, its.prim_id)
        v0 = dr.gather(Array2, self.vertices, f.x)
        v1 = dr.gather(Array2, self.vertices, f.y)
        val0 = dr.gather(type(self.values), self.values, f.x)
        val1 = dr.gather(type(self.values), self.values, f.y)
        dv = (val1 - val0) / dr.norm(v1 - v0)
        d_t = dr.normalize(v1 - v0)
        # d_t = Array2(its.n.y, -its.n.x)  # tangent direction
        return dv * d_t.x, dv * d_t.y

    # io
    def write_obj(self, filename):
        vertices = self.vertices.numpy()
        indices = self.indices.numpy()
        values = self.values.numpy()
        with open(filename, 'w') as f:
            for v in vertices:
                f.write('v %f %f\n' % (v[0], v[1]))

            for i in indices:
                f.write('l %d %d\n' % (i[0] + 1, i[1] + 1))

            for c in values:
                f.write('c %f\n' % c)
