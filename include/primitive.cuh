#include <fwd.cuh>
#include <lbvh.cuh>
#include <util.cuh>
namespace wos_cuda {
template <int N>
struct Primitive;

template <>
struct Primitive<2> {
    Vector2 a;
    Vector2 b;
};

template <>
struct Primitive<3> {
    Vector3 a;
    Vector3 b;
    Vector3 c;
};

template <int N>
struct PrimitiveAABB;

template <>
struct PrimitiveAABB<2> {
    __device__ lbvh::aabb<_float, 2> operator()(const Primitive<2> &p) const {
        lbvh::aabb<_float, 2> aabb;
        aabb.upper = make_float2(max(p.a.x(), p.b.x()), max(p.a.y(), p.b.y()));
        aabb.lower = make_float2(min(p.a.x(), p.b.x()), min(p.a.y(), p.b.y()));
        return aabb;
    }
};

template <>
struct PrimitiveAABB<3> {
    __device__ lbvh::aabb<_float, 3> operator()(const Primitive<3> &p) const {
        lbvh::aabb<_float, 3> aabb;
        aabb.upper = make_float4(
            max(p.a.x(), p.b.x(), p.c.x()),
            max(p.a.y(), p.b.y(), p.c.y()),
            max(p.a.z(), p.b.z(), p.c.z()),
            0.f);
        aabb.lower = make_float4(
            min(p.a.x(), p.b.x(), p.c.x()),
            min(p.a.y(), p.b.y(), p.c.y()),
            min(p.a.z(), p.b.z(), p.c.z()),
            0.f);
        return aabb;
    }
};

template <int N>
struct PrimitiveDistance;

template <>
struct PrimitiveDistance<2> {
    __device__ _float operator()(const float2 &_p, const Primitive<2> &prim) const {
        return distance(Vector2(_p.x, _p.y), prim.a, prim.b);
    }
};

template <>
struct PrimitiveDistance<3> {
    __device__ _float operator()(const float4 &_p, const Primitive<3> &prim) const {
        float d = distance(Vector3(_p.x, _p.y, _p.z), prim.a, prim.b, prim.c);
        return d * d;
    }
};
} // namespace wos_cuda