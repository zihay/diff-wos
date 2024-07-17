from dataclasses import dataclass
from wos.fwd import *
from wos.scene import BoundarySamplingRecord, Detector
from wos.io import write_exr
from wos.utils import closest_point_triangle, interpolate
import wos_ext


@dataclass
class Detector3D(Detector):
    res: tuple = (256, 256)
    z: Float = Float(0.05)
    
    def __post_init__(self):
        self.l_z = (self.vmax[0] - self.vmin[0]) / self.res[0]
    
    def size(self):
        # For boundary sampling, we assume a slab detector instead of a plane.
        return dr.prod(Array2(self.vmax) - Array2(self.vmin)) * self.l_z

    def make_points(self):
        x = dr.linspace(Float, self.vmin[0], self.vmax[0], self.res[0])
        y = dr.linspace(Float, self.vmax[1], self.vmin[1], self.res[1])
        return Array3(dr.meshgrid(x, y, self.z))

    def make_jittered_points(self, sampler, spp):
        p = self.make_points()
        p = dr.repeat(p, spp)
        d = (Array2(self.vmax) - Array2(self.vmin)) / Array2(self.res)
        offset = Array2(sampler.next_float64(),
                        sampler.next_float64()) * d - d / 2.
        p += Array3(offset.x, offset.y, Float(0.))
        return p

    def index(self, p):
        inside_z = dr.abs(p.z - self.z) < (self.l_z / 2.)
        p = Array2(p.x, p.y)
        uv = (p - Array2(self.vmin)) / (Array2(self.vmax) - Array2(self.vmin))
        uv = uv * Array2(self.res[0], self.res[1]) + Array2(0.5, 0.5)
        uv = Array2i(uv)
        valid = (uv.x >= 0) & (uv.x < self.res[0]) & \
                (uv.y >= 0) & (uv.y < self.res[1])
        return inside_z & valid, uv.x + (self.res[1] - uv.y) * self.res[0]

    def save(self, image, filename):
        write_exr(filename, image)

@dataclass
class Detector3DY(Detector):
    res: tuple = (256, 256)
    y: Float = Float(0.05)
    
    def __post_init__(self):
        self.l_y = (self.vmax[0] - self.vmin[0]) / self.res[0]
    
    def size(self):
        # For boundary sampling, we assume a slab detector instead of a plane.
        return dr.prod(Array2(self.vmax) - Array2(self.vmin)) * self.l_z

    def make_points(self):
        x = dr.linspace(Float, self.vmin[0], self.vmax[0], self.res[0])
        z = dr.linspace(Float, self.vmax[1], self.vmin[1], self.res[1])
        return Array3(dr.meshgrid(x, self.y, z))

    def make_jittered_points(self, sampler, spp):
        p = self.make_points()
        p = dr.repeat(p, spp)
        d = (Array2(self.vmax) - Array2(self.vmin)) / Array2(self.res)
        offset = Array2(sampler.next_float64(),
                        sampler.next_float64()) * d - d / 2.
        p += Array3(offset.x, Float(0.), offset.y)
        return p

    def index(self, p):
        inside_y = dr.abs(p.y - self.y) < (self.l_y / 2.)
        p = Array2(p.x, p.z)
        uv = (p - Array2(self.vmin)) / (Array2(self.vmax) - Array2(self.vmin))
        uv = uv * Array2(self.res[0], self.res[1]) + Array2(0.5, 0.5)
        uv = Array2i(uv)
        valid = (uv.x >= 0) & (uv.x < self.res[0]) & \
                (uv.z >= 0) & (uv.z < self.res[1])
        return inside_y & valid, uv.x + (self.res[1] - uv.y) * self.res[0]

    def save(self, image, filename):
        write_exr(filename, image)

@dataclass
class ClosestPointRecord3D:
    valid: Bool = Bool()
    p: Array3 = Array3()
    n: Array3 = Array3()
    uv: Array2 = Array2()
    val: Float = Float()
    prim_id: Int = Int(-1)
    J: Float = Float(1.)

    def c_object(self):
        return wos_ext.ClosestPointRecord3D(self.valid,
                                            self.p,
                                            self.n,
                                            self.uv,
                                            self.prim_id,
                                            Float(0.))


@dataclass
class Scene3D(wos_ext.Scene3D):
    vertices: Array3
    indices: Array3i
    values: Float
    normals: Array3
    use_bvh: bool = False

    edge_distr: mi.DiscreteDistribution = None

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
                elif line.startswith('f '):
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
        vertices = Array3(np.array(vertices))
        indices = Array3i(np.array(indices))
        values = Float(np.array(values))
        return Scene3D(vertices, indices, values)

    def __init__(self, vertices, indices, values, use_bvh=False):
        super().__init__(vertices, indices, values, use_bvh=use_bvh)
        self.init_dist(vertices, indices)

    def init_dist(self, vertices, indices):
        v0 = dr.gather(type(self.vertices), vertices, indices.x)
        v1 = dr.gather(type(self.vertices), vertices, indices.y)
        v2 = dr.gather(type(self.vertices), vertices, indices.z)
        area = dr.norm(dr.cross(v1 - v0, v2 - v0))
        self.edge_distr = mi.DiscreteDistribution(dr.detach(area))
    
    # ! comment out to use c++ implementation
    def closest_point_preliminary(self, p):
        with dr.suspend_grad():
            d = Float(dr.inf)
            idx = Int(-1)
            i = Int(0)
            loop = Loop("closest_point", lambda: (idx, d, i))
            while loop(i < dr.width(self.indices)):
                f = dr.gather(Array3i, self.indices, i)
                a = dr.gather(Array3, self.vertices, f.x)
                b = dr.gather(Array3, self.vertices, f.y)
                c = dr.gather(Array3, self.vertices, f.z)
                pt, uv, _d = closest_point_triangle(p, a, b, c)
                n = dr.normalize(dr.cross(b - a, c - a))
                #! minus sign is important here, helps to break ties
                # _d = -dr.sign(dr.dot(n, p - pt)) * _d
                idx = dr.select(_d < d, i, idx)
                d = dr.select(_d < d, _d, d)
                i += 1
            return idx

    # ! comment out to use c++ implementation
    def closest_point(self, p, active=Bool(True)):
        with dr.suspend_grad():
            idx = self.closest_point_preliminary(p)
        f = dr.gather(Array3i, self.indices, idx, active=active)
        a = dr.gather(Array3, self.vertices, f.x, active=active)
        b = dr.gather(Array3, self.vertices, f.y, active=active)
        c = dr.gather(Array3, self.vertices, f.z, active=active)
        va = dr.gather(type(self.values), self.values, f.x, active=active)
        vb = dr.gather(type(self.values), self.values, f.y, active=active)
        vc = dr.gather(type(self.values), self.values, f.z, active=active)

        _, uv, _ = closest_point_triangle(p, a, b, c)
        pt = interpolate(a, b, c, uv)
        n = dr.normalize(dr.cross(b - a, c - a))
        n *= dr.sign(dr.dot(n, p - pt))
        return ClosestPointRecord3D(valid=active & (idx >= 0),
                                    p=pt,
                                    n=n,
                                    uv=uv,
                                    val=interpolate(va, vb, vc, uv),
                                    prim_id=idx)

    def sdf(self, p: Array3, active=Bool(True)):
        idx = self.closest_point_preliminary(p)
        f = dr.gather(Array3i, self.indices, idx, active=active)
        a = dr.gather(Array3, self.vertices, f.x, active=active)
        b = dr.gather(Array3, self.vertices, f.y, active=active)
        c = dr.gather(Array3, self.vertices, f.z, active=active)
        pt, uv, d = closest_point_triangle(p, a, b, c)
        # the normals is computed in c++ code
        na = dr.gather(Array3, self.normals, f.x)
        nb = dr.gather(Array3, self.normals, f.y)
        nc = dr.gather(Array3, self.normals, f.z)
        n = interpolate(na, nb, nc, uv)
        # n = dr.cross(b - a, c - a)
        return dr.sign(dr.dot(n, p - pt)) * d

    def dirichlet(self, its: ClosestPointRecord3D):
        f = dr.gather(Array3i, self.indices, its.prim_id, active=its.valid)
        va = dr.gather(type(self.values), self.values, f.x, active=its.valid)
        vb = dr.gather(type(self.values), self.values, f.y, active=its.valid)
        vc = dr.gather(type(self.values), self.values, f.z, active=its.valid)
        return dr.select(its.valid,
                         interpolate(va, vb, vc, its.uv),
                         Float(0.))

    def get_point(self, its: ClosestPointRecord3D):
        uv = dr.detach(its.uv)
        f = dr.gather(Array3i, self.indices, its.prim_id, active=its.valid)
        a = dr.gather(Array3, self.vertices, f.x, active=its.valid)
        b = dr.gather(Array3, self.vertices, f.y, active=its.valid)
        c = dr.gather(Array3, self.vertices, f.z, active=its.valid)
        va = dr.gather(type(self.values), self.values, f.x, active=its.valid)
        vb = dr.gather(type(self.values), self.values, f.y, active=its.valid)
        vc = dr.gather(type(self.values), self.values, f.z, active=its.valid)
        return ClosestPointRecord3D(valid=its.valid,
                                    p=interpolate(a, b, c, uv),
                                    n=its.n,
                                    uv=uv,
                                    val=interpolate(va, vb, vc, uv),
                                    prim_id=its.prim_id)

    def sample_boundary(self, sampler):
        rnd = Float(sampler.next_float32())
        prim_id, t, prob = self.edge_distr.sample_reuse_pmf(rnd)
        f = dr.gather(Array3i, self.indices, prim_id)
        a = dr.gather(Array3, self.vertices, f.x)
        b = dr.gather(Array3, self.vertices, f.y)
        c = dr.gather(Array3, self.vertices, f.z)
        va = dr.gather(Float, self.values, f.x)
        vb = dr.gather(Float, self.values, f.y)
        vc = dr.gather(Float, self.values, f.z)
        area = dr.norm(dr.cross(b - a, c - a))
        # uniform sample on triangle
        r1 = Float(sampler.next_float32())
        r2 = Float(sampler.next_float32())
        u = dr.sqrt(r1)*(1 - r2)
        v = dr.sqrt(r1)*r2
        p = interpolate(a, b, c, Array2(u, v))
        val = interpolate(va, vb, vc, Array2(u, v))
        n = -dr.detach(dr.normalize(dr.cross(b - a, c - a)))
        return BoundarySamplingRecord(p=p, n=n, val=val,
                                      prim_id=prim_id, t=Array2(u, v),
                                      pdf=prob / area)

    def write_obj(self, filename):
        vertices = self.vertices.numpy()
        indices = self.indices.numpy()
        values = self.values.numpy()
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for i in indices:
                f.write(f'f {i[0]+1} {i[1]+1} {i[2]+1}\n')
            for v in values:
                f.write(f'c {v}\n')
