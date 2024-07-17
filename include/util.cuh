#pragma once
#include <fwd.cuh>

namespace wos_cuda {
template <int DIM>
struct CreateVector;

template <>
struct CreateVector<2> {
    __device__ Vector2 operator()(thrust::tuple<_float, _float> x) {
        return Vector2(thrust::get<0>(x), thrust::get<1>(x));
    }
};

template <>
struct CreateVector<3> {
    __device__ Vector3 operator()(thrust::tuple<_float, _float, _float> x) {
        return Vector3(thrust::get<0>(x), thrust::get<1>(x), thrust::get<2>(x));
    }
};

inline __device__ float closestPointTriangle(const Vector3 &pa, const Vector3 &pb, const Vector3 &pc,
                                             const Vector3 &x, Vector3 &pt, Vector2 &t) {
    // https://github.com/rohan-sawhney/fcpw/blob/651546533484576b6de212c513e3e0d65f27dea8/include/fcpw/geometry/triangles.inl#L265-L349
    //  source: real time collision detection
    //  check if x in vertex region outside pa
    Vector3 ab = pb - pa;
    Vector3 ac = pc - pa;
    Vector3 ax = x - pa;
    float   d1 = ab.dot(ax);
    float   d2 = ac.dot(ax);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        // barycentric coordinates (1, 0, 0)
        t[0] = 1.0f;
        t[1] = 0.0f;
        pt   = pa;
        return (x - pt).norm();
    }

    // check if x in vertex region outside pb
    Vector3 bx = x - pb;
    float   d3 = ab.dot(bx);
    float   d4 = ac.dot(bx);
    if (d3 >= 0.0f && d4 <= d3) {
        // barycentric coordinates (0, 1, 0)
        t[0] = 0.0f;
        t[1] = 1.0f;
        pt   = pb;
        return (x - pt).norm();
    }

    // check if x in vertex region outside pc
    Vector3 cx = x - pc;
    float   d5 = ab.dot(cx);
    float   d6 = ac.dot(cx);
    if (d6 >= 0.0f && d5 <= d6) {
        // barycentric coordinates (0, 0, 1)
        t[0] = 0.0f;
        t[1] = 0.0f;
        pt   = pc;
        return (x - pt).norm();
    }

    // check if x in edge region of ab, if so return projection of x onto ab
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        // barycentric coordinates (1 - v, v, 0)
        float v = d1 / (d1 - d3);
        t[0]    = 1.0f - v;
        t[1]    = v;
        pt      = pa + ab * v;
        return (x - pt).norm();
    }

    // check if x in edge region of ac, if so return projection of x onto ac
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        // barycentric coordinates (1 - w, 0, w)
        float w = d2 / (d2 - d6);
        t[0]    = 1.0f - w;
        t[1]    = 0.0f;
        pt      = pa + ac * w;
        return (x - pt).norm();
    }

    // check if x in edge region of bc, if so return projection of x onto bc
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        // barycentric coordinates (0, 1 - w, w)
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        t[0]    = 0.0f;
        t[1]    = 1.0f - w;
        pt      = pb + (pc - pb) * w;
        return (x - pt).norm();
    }

    // x inside face region. Compute pt through its barycentric coordinates (u, v, w)
    float denom = 1.0f / (va + vb + vc);
    float v     = vb * denom;
    float w     = vc * denom;
    t[0]        = 1.0f - v - w;
    t[1]        = v;

    pt = pa + ab * v + ac * w; //= u*a + v*b + w*c, u = va*denom = 1.0f - v - w
    return (x - pt).norm();
}

inline __device__ _float distance(const Vector2 &p,
                                  const Vector2 &a, const Vector2 &b) {
    auto pa = p - a;
    auto ba = b - a;
    auto h  = clamp(pa.dot(ba) / ba.dot(ba), 0.0f, 1.0f);
    return (pa - ba * h).norm();
}

inline __device__ _float distance(const Vector3 &p,
                                  const Vector3 &a, const Vector3 &b, const Vector3 &c) {
    Vector3 pt;
    Vector2 uv;
    closestPointTriangle(a, b, c, p, pt, uv);
    Vector3 n = (b - a).cross(c - a);
    Vector3 d = p - pt;
    //! minus sign is important here, helps to break ties
    // return -sign(n.dot(d)) * d.norm();
    return d.norm();
    // // https://iquilezles.org/articles/distfunctions/
    // Vector3 ba  = b - a;
    // Vector3 pa  = p - a;
    // Vector3 cb  = c - b;
    // Vector3 pb  = p - b;
    // Vector3 ac  = a - c;
    // Vector3 pc  = p - c;
    // Vector3 nor = ba.cross(ac);
    // return sqrt(
    //     // some possibilities: 1+1+1=3, -1+1+1=1
    //     (sign(ba.cross(nor).dot(pa)) + // if the projection of p is outside triangle
    //          sign(cb.cross(nor).dot(pb)) +
    //          sign(ac.cross(nor).dot(pc)) <
    //      2.)
    //         ? min(min(
    //                   (ba * clamp(ba.dot(pa) / ba.squaredNorm(), 0.f, 1.f) - pa).squaredNorm(),
    //                   (cb * clamp(cb.dot(pb) / cb.squaredNorm(), 0.f, 1.f) - pb).squaredNorm()),
    //               (ac * clamp(ac.dot(pc) / ac.squaredNorm(), 0.f, 1.f) - pc).squaredNorm())
    //         : nor.dot(pa) * nor.dot(pa) / nor.squaredNorm());
}

inline __device__ Vector2 tuple_to_vector(const thrust::tuple<_float, _float> &x) {
    return Vector2(thrust::get<0>(x), thrust::get<1>(x));
}

inline __device__ Vector3 tuple_to_vector(const thrust::tuple<_float, _float, _float> &x) {
    return Vector3(thrust::get<0>(x), thrust::get<1>(x), thrust::get<2>(x));
}

inline auto drjit_iterator(const dr::Vector2 &x) {
    return thrust::make_zip_iterator(thrust::make_tuple(x.x().begin(), x.y().begin()));
}

inline auto drjit_iterator(const dr::Vector3 &x) {
    return thrust::make_zip_iterator(
        thrust::make_tuple(x.x().begin(), x.y().begin(), x.z().begin()));
}

} // namespace wos_cuda
