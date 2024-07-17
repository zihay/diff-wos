#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fwd.cuh>
#include <lbvh.cuh>
#include <primitive.cuh>
#include <util.cuh>
namespace dr {
/**
 * drjit ClosestPointRecord
 */
template <int DIM>
struct ClosestPointRecord;

template <>
struct ClosestPointRecord<2> {
    static ClosestPointRecord create(const Bool    &valid,
                                     const Vector2 &p, const Vector2 &n,
                                     Float t, Int prim_id,
                                     Float contrib) {
        return ClosestPointRecord{ valid,
                                   p,
                                   n,
                                   t,
                                   prim_id,
                                   contrib };
    }
    Bool    valid;
    Vector2 p;
    Vector2 n;
    Float   t;
    Int     prim_id;
    Float   contrib;
};

template <>
struct ClosestPointRecord<3> {
    static ClosestPointRecord create(const Bool    &valid,
                                     const Vector3 &p, const Vector3 &n,
                                     const Vector2 &uv, const Int &prim_id,
                                     Float contrib) {
        return ClosestPointRecord{ valid,
                                   p,
                                   n,
                                   uv,
                                   prim_id,
                                   contrib };
    }
    Bool    valid;
    Vector3 p;
    Vector3 n;
    Vector2 uv;
    Int     prim_id;
    Float   contrib;
};

}; // namespace dr
namespace wos_cuda {
template <int DIM>
struct AABB {
    Vector<DIM> min;
    Vector<DIM> max;
};

template <int N>
struct ClosestPointRecord;

template <>
struct ClosestPointRecord<2> {
    bool    valid;
    Vector2 p;
    Vector2 n;
    _float  t; // for 2d
    _float  d;
    int     prim_id;
    _float  contrib = 0.;
};

template <>
struct ClosestPointRecord<3> {
    bool    valid;
    Vector3 p;
    Vector3 n = Vector3(0, 1, 0); //! prevent from being NAN
    Vector2 uv;
    _float  d;
    int     prim_id;
    _float  contrib = 0.;
};

//! Device scene
template <int N>
struct SceneDevice {
    using Primitive          = Primitive<N>;
    using PrimitiveDistance  = PrimitiveDistance<N>;
    using Vector             = Vector<N>;
    using Vectori            = Vectori<N>;
    using ClosestPointRecord = ClosestPointRecord<N>;
    using AABB               = AABB<N>;
    __device__ int closestPointPreliminary(const Vector &p) const {
        // max
        _float d       = FLT_MAX;
        int    prim_id = -1;
        for (int i = 0; i < m_n_indices; i++) {
            _float _d = FLT_MAX;
            if constexpr (N == 2) {
                // 2d
                Vectori idx = m_indices[i];
                Vector  a   = m_vertices[idx.x()];
                Vector  b   = m_vertices[idx.y()];
                _d          = distance(p, a, b);
            } else if constexpr (N == 3) {
                // 3d
                Vectori idx = m_indices[i];
                Vector  a   = m_vertices[idx.x()];
                Vector  b   = m_vertices[idx.y()];
                Vector  c   = m_vertices[idx.z()];
                _d          = distance(p, a, b, c);
            }
            if (_d < d) {
                d       = _d;
                prim_id = i;
            }
        }
        return prim_id;
    }

    __device__ int closestPointPreliminaryBVH(const Vector &p) const {
        if constexpr (N == 2) {
            // 2d
            const auto [object_id, sqr_R] =
                lbvh::query_device(m_bvh,
                                   lbvh::nearest(make_float2(p.x(), p.y())),
                                   PrimitiveDistance());
            return object_id;
        } else if constexpr (N == 3) {
            // 3d
            const auto [object_id, sqr_R] =
                lbvh::query_device(m_bvh,
                                   lbvh::nearest(make_float3(p.x(), p.y(), p.z())),
                                   PrimitiveDistance());
            return object_id;
        }
    }

    __device__ _float sdf(const Vector &p) const {
        int idx = -1;
        if (m_use_bvh) {
            idx = closestPointPreliminaryBVH(p);
        } else {
            idx = closestPointPreliminary(p);
        }
        if constexpr (N == 2) {
            Vector2i f  = m_indices[idx];
            Vector2  a  = m_vertices[f.x()];
            Vector2  b  = m_vertices[f.y()];
            Vector2  pa = p - a;
            Vector2  ba = b - a;
            _float   h  = clamp(pa.dot(ba) / ba.dot(ba), 0.0f, 1.0f);
            Vector2  d  = pa - ba * h;
            Vector2  na = m_normals[f.x()];
            Vector2  nb = m_normals[f.y()];
            Vector2  n  = lerp(na, nb, h);
            return d.norm() * sign(n.dot(d));
        } else if constexpr (N == 3) {
            Vector3i f = m_indices[idx];
            Vector3  a = m_vertices[f.x()];
            Vector3  b = m_vertices[f.y()];
            Vector3  c = m_vertices[f.z()];
            Vector2  uv;
            Vector3  x;
            _float   d = closestPointTriangle(a, b, c, p, x, uv);
            Vector3  n = (b - a).cross(c - a).normalized();
            return d * sign(n.dot(p - x));
        }
    }

    __device__ ClosestPointRecord closestPoint(const Vector &p) const {
        int idx = -1;
        if (m_use_bvh) {
            idx = closestPointPreliminaryBVH(p);
        } else {
            idx = closestPointPreliminary(p);
        }
        if constexpr (N == 2) {
            Vector2i f     = m_indices[idx];
            Vector2  a     = m_vertices[f.x()];
            Vector2  b     = m_vertices[f.y()];
            Vector2  pa    = p - a;
            Vector2  ba    = b - a;
            _float   h     = clamp(pa.dot(ba) / ba.dot(ba), 0.0f, 1.0f);
            _float   d     = (pa - ba * h).norm();
            Vector2  its_n = Vector2(ba.y(), -ba.x()).normalized(); //! outward normal
            its_n          = sign(its_n.dot(pa)) * its_n;

            return ClosestPointRecord{ true,
                                       lerp(a, b, h),
                                       its_n,
                                       h,
                                       d,
                                       idx,
                                       0.f };
        } else if constexpr (N == 3) {
            Vector3i f = m_indices[idx];
            Vector3  a = m_vertices[f.x()];
            Vector3  b = m_vertices[f.y()];
            Vector3  c = m_vertices[f.z()];
            Vector2  uv;
            Vector3  x;
            _float   d     = closestPointTriangle(a, b, c, p, x, uv);
            Vector3  its_n = (b - a).cross(c - a).normalized();
            its_n          = sign(its_n.dot(p - a)) * its_n;
            return ClosestPointRecord{ true,
                                       x,
                                       its_n,
                                       Vector2(uv.y(), 1. - uv.x() - uv.y()),
                                       d,
                                       idx,
                                       0.f };
        }
    }

    __device__ _float largestInscribedBall(const ClosestPointRecord &cRec, _float epsilon = 1e-3) const {
        if (!cRec.valid)
            return 0.;
        // binary search
        _float b = (m_aabb.max - m_aabb.min).cwiseAbs().minCoeff() / 2.;
        _float a = epsilon;
        for (int i = 0; i < 10; i++) {
            _float m = (a + b) / 2.;
            _float d = abs(sdf(cRec.p + cRec.n * m));
            if (abs(d - m) < 1e-5)
                a = m;
            else
                b = m;
        }
        return a;
        // _float r = (m_aabb.max - m_aabb.min).cwiseAbs().minCoeff() / 2.;
        // Vector p = cRec.p + cRec.n * r;
        // _float d = abs(sdf(p));
        // while (abs(d - r) > 1e-6) {
        //     r /= 2.;
        //     p = cRec.p + cRec.n * r;
        //     d = abs(sdf(p));
        // }
        // return r;
    }

    __device__ _float dirichlet(const ClosestPointRecord &cRec) const {
        if constexpr (N == 2) {
            Vector2i f = m_indices[cRec.prim_id];
            _float   a = m_values[f.x()];
            _float   b = m_values[f.y()];
            return lerp(a, b, cRec.t);
        } else if constexpr (N == 3) {
            Vector3i f = m_indices[cRec.prim_id];
            _float   a = m_values[f.x()];
            _float   b = m_values[f.y()];
            _float   c = m_values[f.z()];
            return interpolate(a, b, c, cRec.uv);
        }
    }

    __device__ bool has_source() const {
        return m_source_type != 0;
    }

    /*
     * This implementation should be consistent with the Python version in `scene.py`.
     * source_type: 
     *      0 - none.
     *      1 - Gaussian: params = [mu (2 floats), sigma (2 floats), amplitude (1 float)]
     *      2 - sinusoid: params = [A (1 float), B (1 float), C (1 float), amplitude (1 float), offset (1 float)]
     *
    */
    __device__ _float source_function(const Vector &p) const {
        if (m_source_type == 0) {
            return 0.;
        } else if (m_source_type == 1) {
            _float mu_x = m_source_params[0];
            _float mu_y = m_source_params[1];
            _float sigma_x = m_source_params[2];
            _float sigma_y = m_source_params[3];
            _float A = m_source_params[4];            
            _float dx = (p.x() - mu_x) * (p.x() - mu_x) / (2.f * sigma_x * sigma_x);
            _float dy = (p.y() - mu_y) * (p.y() - mu_y) / (2.f * sigma_y * sigma_y);
            return A * exp(-dx - dy);
        } else if (m_source_type == 2) {
            _float A = m_source_params[0];
            _float B = m_source_params[1];
            _float C = m_source_params[2];
            _float amplitude = m_source_params[3];
            _float offset = m_source_params[4];
            _float phi = A * p.x() + B * p.y() + C;
            return amplitude * sin(phi) + offset;
        } else {
            return 0.;
        }
    }

    Vector                                *m_vertices;
    Vector                                *m_normals;
    Vectori                               *m_indices;
    _float                                *m_values;
    size_t                                 m_n_vertices;
    size_t                                 m_n_indices;
    AABB                                   m_aabb;
    lbvh::bvh_device<_float, N, Primitive> m_bvh;
    bool                                   m_use_bvh = false;
    int                                    m_source_type;
    _float                                *m_source_params;    
};

//! Host scene
template <int N>
class Scene {
    using Primitive     = Primitive<N>;
    using PrimitiveAABB = PrimitiveAABB<N>;
    using SceneDevice   = SceneDevice<N>;
    using drVector      = dr::Vector<N>;
    using drVectori     = dr::Vectori<N>;
    using Vector        = Vector<N>;
    using Vectori       = Vectori<N>;
    using AABB          = AABB<N>;

public:
    Scene() {}
    Scene(const drVector &vertices, const drVectori &indices, const dr::Float &values, bool use_bvh = false)
        : m_vertices(vertices), m_indices(indices), m_values(values), m_use_bvh(use_bvh) {
        configure_normal();
        configure();
    };
    Scene(const drVector &vertices, const drVectori &indices, const dr::Float &values, bool use_bvh,
          int source_type, const dr::Float &source_params)
        : m_vertices(vertices), m_indices(indices), m_values(values), m_use_bvh(use_bvh),
          m_source_type(source_type), m_source_params(source_params) {
        configure_normal();
        configure_source();
        configure();
    };

    static Scene fromVertices(const drVector &vertices, const dr::Float &values) {
        if constexpr (N == 2) {
            dr::Int      i       = drjit::arange<dr::Int>(drjit::width(vertices));
            dr::Vector2i indices = dr::Vector2i(i, i + 1 % drjit::width(vertices));
            return Scene(vertices, indices, values);
        } else {
            throw std::runtime_error("not implemented");
        }
    }

    SceneDevice device() {
        return SceneDevice{ m_vertices_device.data().get(),
                            m_normals_device.data().get(),
                            m_indices_device.data().get(),
                            m_values_device.data().get(),
                            m_vertices_device.size(),
                            m_indices_device.size(),
                            m_aabb,
                            m_bvh.get_device_repr(),
                            m_use_bvh,
                            m_source_type,
                            m_source_params_device.data().get() };
    }

    void configure_normal() {
        if constexpr (N == 2) {
            dr::Vector2 p0 = drjit::gather<dr::Vector2>(m_vertices, m_indices.x());
            dr::Vector2 p1 = drjit::gather<dr::Vector2>(m_vertices, m_indices.y());
            dr::Vector2 d  = p1 - p0;
            dr::Vector2 n  = drjit::normalize(dr::Vector2(d.y(), -d.x()));
            m_normals      = drjit::zeros<dr::Vector2>(drjit::width(m_vertices));
            //! caution: the target and source need to be Float array, and indices need to be an IntAD array.
            drjit::scatter_reduce(ReduceOp::Add, m_normals.x(), n.x(), m_indices.x());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.y(), n.y(), m_indices.x());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.x(), n.x(), m_indices.y());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.y(), n.y(), m_indices.y());
            m_normals = drjit::normalize(m_normals);
        } else if constexpr (N == 3) {
            dr::Vector3 p0 = drjit::gather<dr::Vector3>(m_vertices, m_indices.x());
            dr::Vector3 p1 = drjit::gather<dr::Vector3>(m_vertices, m_indices.y());
            dr::Vector3 p2 = drjit::gather<dr::Vector3>(m_vertices, m_indices.z());
            // face normal
            dr::Vector3 n = drjit::normalize(drjit::cross(p1 - p0, p2 - p0));
            // vertex normal
            m_normals  = drjit::zeros<dr::Vector3>(drjit::width(m_vertices));
            auto angle = [](const dr::Vector3 &a, const dr::Vector3 &b) {
                dr::Float s = drjit::norm(drjit::cross(a, b));
                dr::Float c = drjit::dot(a, b);
                return drjit::atan2(s, c);
            };
            dr::Float angleA = angle(p1 - p0, p2 - p0);
            dr::Float angleB = angle(p2 - p1, p0 - p1);
            dr::Float angleC = angle(p0 - p2, p1 - p2);
            //! caution: the target and source need to be Float array, and indices need to be an IntAD array.
            drjit::scatter_reduce(ReduceOp::Add, m_normals.x(), angleA * n.x(), m_indices.x());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.y(), angleA * n.y(), m_indices.x());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.z(), angleA * n.z(), m_indices.x());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.x(), angleB * n.x(), m_indices.y());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.y(), angleB * n.y(), m_indices.y());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.z(), angleB * n.z(), m_indices.y());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.x(), angleC * n.x(), m_indices.z());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.y(), angleC * n.y(), m_indices.z());
            drjit::scatter_reduce(ReduceOp::Add, m_normals.z(), angleC * n.z(), m_indices.z());
            m_normals = drjit::normalize(m_normals);
        }
    }

    void configure_source() {
        drjit::eval(m_source_params);
        m_source_params_device = to_device_vector(m_source_params);
    }

    //! convert drjit array to device_vector. SoA -> AoS
    void configure() {
        drjit::eval(m_vertices, m_normals, m_indices, m_values);
        //! SoA -> AoS
        m_vertices_device = to_device_vector(m_vertices);
        m_normals_device  = to_device_vector(m_normals);
        m_indices_device  = to_device_vector(m_indices);
        m_values_device   = to_device_vector(m_values);

        thrust::host_vector<Vector>  vertices_host = m_vertices_device;
        thrust::host_vector<Vectori> indices_host  = m_indices_device;

        // build aabb
        m_aabb.min = Vector::Constant(FLT_MAX);
        m_aabb.max = Vector::Constant(-FLT_MAX);
        for (int i = 0; i < vertices_host.size(); i++) {
            Vector v   = vertices_host[i];
            m_aabb.min = m_aabb.min.cwiseMin(v);
            m_aabb.max = m_aabb.max.cwiseMax(v);
        }

        // build bvh
        std::vector<Primitive> primitives;
        for (int i = 0; i < indices_host.size(); i++) {
            Vectori idx = indices_host[i];
            if constexpr (N == 2) {
                Vector2 a = vertices_host[idx.x()];
                Vector2 b = vertices_host[idx.y()];
                primitives.push_back(Primitive{ a, b });
            } else if constexpr (N == 3) {
                Vector3 a = vertices_host[idx.x()];
                Vector3 b = vertices_host[idx.y()];
                Vector3 c = vertices_host[idx.z()];
                primitives.push_back(Primitive{ a, b, c });
            }
        }
        m_bvh = lbvh::bvh<_float, N, Primitive, PrimitiveAABB>(primitives.begin(), primitives.end());
    }

    dr::Float dirichlet(const dr::ClosestPointRecord<N> &cRec) {
        if constexpr (N == 2) {
            dr::Vector2i f = drjit::gather<dr::Vector2i>(m_indices, cRec.prim_id);
            dr::Float    a = drjit::gather<dr::Float>(m_values, f.x());
            dr::Float    b = drjit::gather<dr::Float>(m_values, f.y());
            return drjit::lerp(a, b, cRec.t);
        } else if constexpr (N == 3) {
            dr::Vector3i f = drjit::gather<dr::Vector3i>(m_indices, cRec.prim_id);
            dr::Float    a = drjit::gather<dr::Float>(m_values, f.x());
            dr::Float    b = drjit::gather<dr::Float>(m_values, f.y());
            dr::Float    c = drjit::gather<dr::Float>(m_values, f.z());
            return b * cRec.uv.x() + c * cRec.uv.y() + a * (1 - cRec.uv.x() - cRec.uv.y());
        }
    }

    //! export
    dr::Int closestPointPreliminary(const drVector &p);

    //! export
    dr::ClosestPointRecord<N> closestPoint(const drVector &p);

    //! export
    dr::Float largestInscribedBall(const dr::ClosestPointRecord<N> &cRec, _float epsilon = 1e-3);

    drVector  m_vertices;
    drVector  m_normals;
    drVectori m_indices;
    dr::Float m_values;
    dr::Float m_source_params;
    //! store device vector to prevent from being freed.
    thrust::device_vector<Vector>  m_vertices_device;
    thrust::device_vector<Vector>  m_normals_device;
    thrust::device_vector<Vectori> m_indices_device;
    thrust::device_vector<_float>  m_values_device;
    thrust::device_vector<_float>  m_source_params_device;
    // acceleration structure
    AABB                                           m_aabb;
    lbvh::bvh<_float, N, Primitive, PrimitiveAABB> m_bvh;
    bool                                           m_use_bvh = false;
    int                                            m_source_type;
};

template <>
inline dr::Int Scene<2>::closestPointPreliminary(const drVector &p) {
    drjit::eval(p);
    size_t  npoints = drjit::width(p);
    dr::Int result  = drjit::zeros<dr::Int>(npoints);
    drjit::make_opaque(result);
    void *stream = jit_cuda_stream();
    //! it is important to convert to a device struct. since the lambda function cannot capture this pointer.
    SceneDevice scene = device();

    auto begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(p[0].begin(), p[1].begin())),
        CreateVector<2>());
    thrust::transform(thrust::device.on((cudaStream_t) stream),
                      begin, begin + npoints, result.begin(),
                      [scene] __device__(const Vector2 &p) {
                          return scene.closestPointPreliminary(p);
                      });
    return result;
}

template <>
inline dr::Int Scene<3>::closestPointPreliminary(const drVector &p) {
    drjit::eval(p);
    size_t  npoints = drjit::width(p);
    dr::Int result  = drjit::zeros<dr::Int>(npoints);
    drjit::make_opaque(result);
    void *stream = jit_cuda_stream();
    //! it is important to convert to a device struct. since the lambda function cannot capture this pointer.
    SceneDevice scene = device();

    auto begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(p[0].begin(), p[1].begin(), p[2].begin())),
        CreateVector<3>());
    thrust::transform(thrust::device.on((cudaStream_t) stream),
                      begin, begin + npoints, result.begin(),
                      [scene] __device__(const Vector3 &p) {
                          return scene.closestPointPreliminary(p);
                      });
    return result;
}

template <>
inline dr::ClosestPointRecord<2> Scene<2>::closestPoint(const drVector &p) {
    drjit::eval(p);
    int   npoints = drjit::width(p);
    void *stream  = jit_cuda_stream();
    //! it is important to convert to a device struct. since the lambda function cannot capture this pointer.
    SceneDevice scene        = device();
    dr::Bool    cRec_valid   = drjit::zeros<dr::Bool>(npoints);
    drVector    cRec_p       = drjit::zeros<drVector>(npoints);
    drVector    cRec_n       = drjit::zeros<drVector>(npoints);
    dr::Float   cRec_t       = drjit::zeros<dr::Float>(npoints);
    dr::Vector2 cRec_uv      = drjit::zeros<dr::Vector2>(npoints);
    dr::Int     cRec_prim_id = drjit::zeros<dr::Int>(npoints);
    dr::Float   cRec_contrib = drjit::zeros<dr::Float>(npoints);
    //! important. otherwise, they will point to the same memory.
    drjit::make_opaque(cRec_valid);
    drjit::make_opaque(cRec_p);
    drjit::make_opaque(cRec_n);
    drjit::make_opaque(cRec_t);
    drjit::make_opaque(cRec_uv);
    drjit::make_opaque(cRec_prim_id);
    drjit::make_opaque(cRec_contrib);
    auto input_begin  = thrust::make_zip_iterator(thrust::make_tuple(p.x().begin(), p.y().begin()));
    auto output_begin = thrust::make_zip_iterator(thrust::make_tuple(cRec_valid.begin(),
                                                                     cRec_p.x().begin(), cRec_p.y().begin(),
                                                                     cRec_n.x().begin(), cRec_n.y().begin(),
                                                                     cRec_t.begin(), cRec_prim_id.begin(),
                                                                     cRec_contrib.begin()));
    auto begin        = thrust::make_zip_iterator(thrust::make_tuple(input_begin,
                                                                     output_begin));
    //! AoS -> drjit SoA
    thrust::for_each(thrust::device.on((cudaStream_t) stream),
                     begin, begin + npoints,
                     [scene] __device__(auto t) {
                         auto &input            = thrust::get<0>(t);
                         auto &p_x              = thrust::get<0>(input);
                         auto &p_y              = thrust::get<1>(input);
                         auto  cRec             = scene.closestPoint(Vector2(p_x, p_y));
                         auto &output           = thrust::get<1>(t);
                         thrust::get<0>(output) = cRec.valid;
                         thrust::get<1>(output) = cRec.p.x();
                         thrust::get<2>(output) = cRec.p.y();
                         thrust::get<3>(output) = cRec.n.x();
                         thrust::get<4>(output) = cRec.n.y();
                         thrust::get<5>(output) = cRec.t;
                         thrust::get<6>(output) = cRec.prim_id;
                         thrust::get<7>(output) = cRec.contrib;
                     });
    return dr::ClosestPointRecord<2>{ cRec_valid,
                                      cRec_p,
                                      cRec_n,
                                      cRec_t,
                                      cRec_prim_id,
                                      cRec_contrib };
}

template <>
inline dr::ClosestPointRecord<3> Scene<3>::closestPoint(const drVector &p) {
    drjit::eval(p);
    int   npoints = drjit::width(p);
    void *stream  = jit_cuda_stream();
    //! it is important to convert to a device struct. since the lambda function cannot capture this pointer.
    SceneDevice scene        = device();
    dr::Bool    cRec_valid   = drjit::zeros<dr::Bool>(npoints);
    drVector    cRec_p       = drjit::zeros<drVector>(npoints);
    drVector    cRec_n       = drjit::zeros<drVector>(npoints);
    dr::Float   cRec_t       = drjit::zeros<dr::Float>(npoints);
    dr::Vector2 cRec_uv      = drjit::zeros<dr::Vector2>(npoints);
    dr::Int     cRec_prim_id = drjit::zeros<dr::Int>(npoints);
    dr::Float   cRec_contrib = drjit::zeros<dr::Float>(npoints);
    //! important. otherwise, they will point to the same memory.
    drjit::make_opaque(cRec_valid);
    drjit::make_opaque(cRec_p);
    drjit::make_opaque(cRec_n);
    drjit::make_opaque(cRec_t);
    drjit::make_opaque(cRec_uv);
    drjit::make_opaque(cRec_prim_id);
    drjit::make_opaque(cRec_contrib);
    auto input_begin  = thrust::make_zip_iterator(thrust::make_tuple(p.x().begin(), p.y().begin(), p.z().begin()));
    auto output_begin = thrust::make_zip_iterator(thrust::make_tuple(cRec_valid.begin(),
                                                                     cRec_p.x().begin(), cRec_p.y().begin(), cRec_p.z().begin(),
                                                                     cRec_n.x().begin(), cRec_n.y().begin(), cRec_n.z().begin(),
                                                                     cRec_uv.x().begin(), cRec_uv.y().begin(),
                                                                     cRec_prim_id.begin()));
    auto output_begin2 = thrust::make_zip_iterator(thrust::make_tuple(cRec_contrib.begin()));

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(input_begin,
                                                              output_begin,
                                                              output_begin2));
    //! AoS -> drjit SoA
    thrust::for_each(thrust::device.on((cudaStream_t) stream),
                     begin, begin + npoints,
                     [scene] __device__(auto t) {
                         auto &input            = thrust::get<0>(t);
                         auto &p_x              = thrust::get<0>(input);
                         auto &p_y              = thrust::get<1>(input);
                         auto &p_z              = thrust::get<2>(input);
                         auto  cRec             = scene.closestPoint(Vector3(p_x, p_y, p_z));
                         auto &output           = thrust::get<1>(t);
                         auto &output2          = thrust::get<2>(t);
                         thrust::get<0>(output) = cRec.valid;
                         thrust::get<1>(output) = cRec.p.x();
                         thrust::get<2>(output) = cRec.p.y();
                         thrust::get<3>(output) = cRec.p.z();
                         thrust::get<4>(output) = cRec.n.x();
                         thrust::get<5>(output) = cRec.n.y();
                         thrust::get<6>(output) = cRec.n.z();
                         thrust::get<7>(output) = cRec.uv.x();
                         thrust::get<8>(output) = cRec.uv.y();
                         thrust::get<9>(output) = cRec.prim_id;                         
                         thrust::get<0>(output2) = cRec.contrib;
                     });
    return dr::ClosestPointRecord<3>{ cRec_valid,
                                      cRec_p,
                                      cRec_n,
                                      cRec_uv,
                                      cRec_prim_id,
                                      cRec_contrib };
}

template <>
inline dr::Float Scene<2>::largestInscribedBall(const dr::ClosestPointRecord<2> &cRec, _float epsilon) {
    drjit::eval(cRec.p, cRec.n);
    size_t    npoints = drjit::width(cRec.p);
    dr::Float r       = drjit::zeros<dr::Float>(npoints);
    void     *stream  = jit_cuda_stream();
    //! it is important to convert to a device struct. since the lambda function cannot capture `this` pointer.
    SceneDevice scene = device();
    auto        begin = thrust::zip_iterator(thrust::make_tuple(cRec.valid.begin(),
                                                                cRec.p.x().begin(), cRec.p.y().begin(),
                                                                cRec.n.x().begin(), cRec.n.y().begin()));
    thrust::transform(thrust::device.on((cudaStream_t) stream), begin, begin + npoints, r.begin(),
                      [epsilon, scene] __device__(thrust::tuple<bool, float, float, float, float> t) {
                          auto    valid = thrust::get<0>(t);
                          Vector2 p(thrust::get<1>(t), thrust::get<2>(t));
                          Vector2 n(thrust::get<3>(t), thrust::get<4>(t));
                          auto    cRec = ClosestPointRecord<2>{ valid, p, n };
                          return scene.largestInscribedBall(cRec, epsilon);
                      });
    return r;
}

template <>
inline dr::Float Scene<3>::largestInscribedBall(const dr::ClosestPointRecord<3> &cRec, _float epsilon) {
    drjit::eval(cRec.p, cRec.n);
    size_t    npoints = drjit::width(cRec.p);
    dr::Float r       = drjit::zeros<dr::Float>(npoints);
    void     *stream  = jit_cuda_stream();
    //! it is important to convert to a device struct. since the lambda function cannot capture `this` pointer.
    SceneDevice scene = device();
    auto        begin = thrust::zip_iterator(thrust::make_tuple(cRec.valid.begin(),
                                                                cRec.p.x().begin(), cRec.p.y().begin(), cRec.p.z().begin(),
                                                                cRec.n.x().begin(), cRec.n.y().begin(), cRec.n.z().begin()));
    thrust::transform(thrust::device.on((cudaStream_t) stream), begin, begin + npoints, r.begin(),
                      [epsilon, scene] __device__(thrust::tuple<bool,
                                                                float, float, float,
                                                                float, float, float>
                                                      t) {
                          auto    valid = thrust::get<0>(t);
                          Vector3 p(thrust::get<1>(t), thrust::get<2>(t), thrust::get<3>(t));
                          Vector3 n(thrust::get<4>(t), thrust::get<5>(t), thrust::get<6>(t));
                          auto    cRec = ClosestPointRecord<3>{ valid, p, n };
                          return scene.largestInscribedBall(cRec, epsilon);
                      });
    return r;
}

} // namespace wos_cuda