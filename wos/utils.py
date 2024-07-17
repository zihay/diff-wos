from typing import List
from wos.fwd import *


def lerp(a, b, t):
    return a * (1 - t) + b * t


def bilinear(a, b, c, d, u, v):
    return lerp(lerp(a, b, u), lerp(c, d, u), v)


def interpolate(a, b, c, uv):
    return b * uv[0] + c * uv[1] + a * (1 - uv[0] - uv[1])


def ray_segment_intersect(o: Array2, d: Array2, a: Array2, b: Array2, _active=Bool(True)):
    v1 = o - a
    v2 = b - a
    v3 = Array2(-d.y, d.x)
    dot = dr.dot(v2, v3)
    is_parallel = dr.abs(dot) < EPSILON
    active = _active & ~is_parallel
    t_ray = (v2.x * v1.y - v2.y * v1.x) / dot
    t_seg = dr.dot(v1, v3) / dot
    is_hit = (t_ray > 0) & (t_seg >= 0.) & (t_seg <= 1.)
    active &= is_hit
    is_hit = active
    t = dr.select(active, t_ray, dr.inf)
    n = dr.normalize(Array2(v2.y, -v2.x))
    p = a + (b - a) * t_seg
    t_ray = dr.select(active, t_ray, dr.inf)
    return is_hit, t_ray, t_seg, p, n


def ray_segments_intersect(o: Array2, d: Array2, segments: List):
    t_ray = dr.inf
    t_seg = dr.inf
    idx = Int(-1)
    is_hit = Bool(False)
    p = Array2(0., 0.)
    n = Array2(0., 0.)
    for i, segment in enumerate(segments):
        _is_hit, _t_ray, _t_seg, _p, _n = ray_segment_intersect(
            o, d, segment[0], segment[1])
        is_hit = is_hit | _is_hit
        active = _t_ray < t_ray
        t_ray = dr.select(active, _t_ray, t_ray)
        idx = dr.select(active, i, idx)
        t_seg = dr.select(active, _t_seg, t_seg)
        p = dr.select(active, _p, p)
        n = dr.select(active, _n, n)
    return is_hit, idx, t_ray, t_seg, p, n


def ray_lines_intersect(o: Array2, d: Array2, V: Array2, F: Array2i):
    t_ray = dr.inf
    t_seg = dr.inf
    idx = Int(-1)
    is_hit = Bool(False)
    p = Array2(0., 0.)
    n = Array2(0., 0.)
    for i in range(dr.width(F)):
        i0, i1 = dr.gather(Array2i, F, i)
        v0, v1 = dr.gather(Array2, V, i0), dr.gather(Array2, V, i1)
        _is_hit, _t_ray, _t_seg, _p, _n = ray_segment_intersect(o, d, v0, v1)
        is_hit = is_hit | _is_hit
        active = _t_ray < t_ray
        t_ray = dr.select(active, _t_ray, t_ray)
        idx = dr.select(active, i, idx)
        t_seg = dr.select(active, _t_seg, t_seg)
        p = dr.select(active, _p, p)
        n = dr.select(active, _n, n)
    return is_hit, idx, t_ray, t_seg, p, n

# When there are multiple intersections, we randomly choose one using Weighted Reservoir Sampling


def ray_lines_intersect_all(o: Array2, d: Array2, V: Array2, F: Array2i,
                            sampler: PCG32):
    t_ray = dr.inf
    t_seg = dr.inf
    idx = Int(-1)
    is_hit = Bool(False)
    p = Array2(0., 0.)
    n = Array2(0., 0.)

    count = Int(0)

    for i in range(dr.width(F)):
        i0, i1 = dr.gather(Array2i, F, i)
        v0, v1 = dr.gather(Array2, V, i0), dr.gather(Array2, V, i1)
        _is_hit, _t_ray, _t_seg, _p, _n = ray_segment_intersect(o, d, v0, v1)

        # Weighted Reservoir Sampling
        count[_is_hit] += 1.
        active = _is_hit & (sampler.next_float64() < (1. / Float(count)))

        is_hit = is_hit | _is_hit
        t_ray = dr.select(active, _t_ray, t_ray)
        idx = dr.select(active, i, idx)
        t_seg = dr.select(active, _t_seg, t_seg)
        p = dr.select(active, _p, p)
        n = dr.select(active, _n, n)
    return is_hit, idx, t_ray, t_seg, p, n, count


def closest_point_triangle(p: Array3, a: Array3, b: Array3, c: Array3):
    pt = Array3(0, 0, 0)
    uv = Array2(0, 0)
    d = dr.inf
    ab = b - a
    ac = c - a
    active = Bool(True)
    # check if p is in the vertex region outside a
    ax = p - a
    d1 = dr.dot(ab, ax)
    d2 = dr.dot(ac, ax)
    cond = (d1 <= 0) & (d2 <= 0)
    pt = dr.select(cond, a, pt)
    uv = dr.select(cond, Array2(1, 0), uv)
    d = dr.select(cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the vertex region outside b
    bx = p - b
    d3 = dr.dot(ab, bx)
    d4 = dr.dot(ac, bx)
    cond = (d3 >= 0) & (d4 <= d3)
    pt = dr.select(active & cond, b, pt)
    uv = dr.select(active & cond, Array2(0, 1), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the vertex region outside c
    cx = p - c
    d5 = dr.dot(ab, cx)
    d6 = dr.dot(ac, cx)
    cond = (d6 >= 0) & (d5 <= d6)
    pt = dr.select(active & cond, c, pt)
    uv = dr.select(active & cond, Array2(0, 0), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of ab, if so return projection of p onto ab
    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    cond = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    pt = dr.select(active & cond, a + ab * v, pt)
    uv = dr.select(active & cond, Array2(1 - v, v), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of ac, if so return projection of p onto ac
    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    cond = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    pt = dr.select(active & cond, a + ac * w, pt)
    uv = dr.select(active & cond, Array2(1 - w, 0), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of bc, if so return projection of p onto bc
    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    cond = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    pt = dr.select(active & cond, b + (c - b) * w, pt)
    uv = dr.select(active & cond, Array2(0, 1 - w), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is inside face region
    denom = 1. / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    pt = dr.select(active, a + ab * v + ac * w, pt)
    uv = dr.select(active, Array2(1 - v - w, v), uv)
    d = dr.select(active, dr.norm(p - pt), d)
    return pt, Array2(uv[1], 1. - uv[0] - uv[1]), d


def rand_on_circle(sampler):
    u = sampler.next_float64()
    theta = 2. * dr.pi * u
    return Array2(dr.cos(theta), dr.sin(theta))


def rand_on_sphere(sampler):
    u = sampler.next_float64()
    v = sampler.next_float64()
    theta = 2. * dr.pi * u
    phi = dr.acos(2. * v - 1.)
    return Array3(dr.sin(phi) * dr.cos(theta),
                  dr.sin(phi) * dr.sin(theta),
                  dr.cos(phi))


def rotate(v, angle):
    c = dr.cos(angle)
    s = dr.sin(angle)
    return Matrix2([[c, -s], [s, c]]) @ v


def rotate_axis(v, axis, angle):
    m = mi.Transform4f.rotate(axis, angle/dr.pi*180.)
    return Array3(m @ v)


def rotate_euler(v, euler):
    Q = dr.euler_to_quat(Array3(euler))
    m = dr.quat_to_matrix(Q, size=3)
    return m @ v


def translate(v, t):
    return v + t


def scale(v, s):
    return v * s


def concat(a, b):
    assert (type(a) == type(b))
    size_a = dr.width(a)
    size_b = dr.width(b)
    c = dr.empty(type(a), size_a + size_b)
    dr.scatter(c, a, dr.arange(Int, size_a))
    dr.scatter(c, b, size_a + dr.arange(Int, size_b))
    return c


def meshgrid(vmin=[-1., -1.], vmax=[1., 1.], n=100):
    x = dr.linspace(Float, vmin[0], vmax[1], n)
    y = dr.linspace(Float, vmin[0], vmax[1], n)
    return Array2(dr.meshgrid(x, y))


def plot_lorenz_curve(data):
    import matplotlib.pyplot as plt
    data = np.sort(data)
    y = data.cumsum() / data.sum()
    x = np.arange(len(data)) / len(data)
    plt.plot(x, y)


def plot_ci(data, **kwargs):
    import matplotlib.pyplot as plt
    from wos.stats import Statistics
    stats = Statistics()
    m = stats.mean(data)
    ci = stats.ci(data)
    plt.plot(m, **kwargs)
    plt.fill_between(np.arange(len(m)), m-ci, m+ci, alpha=0.3)

def sample_tea_32(v0, v1, round=4):
    sum = Int(0)
    for i in range(round):
        sum += 0x9e3779b9
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4)
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e)
    return v0, v1