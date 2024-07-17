#pragma once
// #include <sampler.cuh>
#include <sampler2.cuh>
#include <scene.cuh>
#include <solver.cuh>
#include <tabulated_G_cdf.cuh>
#include <util.cuh>
namespace wos_cuda {
template <int N>
struct WoSDevice {
    using SceneDevice        = SceneDevice<N>;
    using Vector             = Vector<N>;
    using ClosestPointRecord = ClosestPointRecord<N>;
    __device__ Vector sample_uniform(Sampler &sampler) const {
        if constexpr (N == 2) {
            _float rnd   = sampler.next_1d();
            _float angle = 2. * M_PI * rnd;
            return Vector2(cos(angle), sin(angle));
        } else if constexpr (N == 3) {
            auto  u   = sampler.next_2d();
            float z   = 1.0f - 2.0f * u[0];
            float r   = sqrt(max(0.0f, 1.0f - z * z));
            float phi = 2.0f * M_PI * u[1];
            return Vector3(r * cos(phi), r * sin(phi), z);
        }
    }

    __device__ Vector sample_uniform_inside_ball(Sampler &sampler) const {
        if constexpr (N == 2) {
            auto   rnd   = sampler.next_2d();
            _float angle = 2. * M_PI * rnd[0];
            _float r     = sqrt(rnd[1]);
            return Vector2(r * cos(angle), r * sin(angle));
        } else if constexpr (N == 3) {
            auto  u         = sampler.next_3d();
            float cos_theta = 1.f - 2.f * u[0];
            float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
            float phi       = 2.0f * M_PI * u[1];
            float r         = pow(u[2], 1.f / 3.f);
            return Vector3(r * sin_theta * cos(phi), r * sin_theta * sin(phi), r * cos_theta);
        }
    }

    __device__ _float eval_G_2D(const Vector &p, _float R, _float r_clamp = 1e-4f) const {
        _float r = max(p.norm(), r_clamp);
        return log(R / r) / (2. * M_PI);
    }

    __device__ _float sample_G_radius(_float u, _float R, _float &pdf) const {
        int l = 0, r = G_cdf_size - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (u < G_cdf[mid])
                r = mid;
            else
                l = mid + 1;
        }
        l = clamp(l, 0, G_cdf_size - 1);

        _float pmf, cdf_prev;
        if (l == 0) {
            pmf      = G_cdf[0];
            cdf_prev = 0.0;
        } else {
            pmf      = G_cdf[l] - G_cdf[l - 1];
            cdf_prev = G_cdf[l - 1];
        }

        // reuse sample
        u = (u - cdf_prev) / pmf;

        _float radius = R * (l + u) / _float(G_cdf_size);
        pdf           = pmf * _float(G_cdf_size) / R;
        return radius;
    }

    __device__ Vector sample_G_2D(Sampler &sampler, _float R, _float &pdf) const {
        if constexpr (N == 2) {
            auto   u     = sampler.next_2d();
            _float r     = sample_G_radius(u[0], R, pdf);
            _float angle = 2. * M_PI * u[1];
            pdf /= 2.0 * M_PI;
            return Vector2(r * cos(angle), r * sin(angle));
        } else if constexpr (N == 3) {
            // TODO
            return Vector3(0.0, 0.0, 1.0);
        }
    }

    __device__ _float u(const Vector &_p, const SceneDevice &scene, Sampler &sampler) const {
        auto   its    = walk(_p, scene, sampler);
        _float result = its.valid ? scene.dirichlet(its) : 0.;
        result += scene.has_source() ? its.contrib : 0.;
        return result;
    }

    __device__ ClosestPointRecord walk(const Vector &_p, const SceneDevice &scene, Sampler &sampler) const {
        Vector p(_p);
        _float contrib = 0.f;
        _float d       = scene.sdf(p);
        //! d > 0 and d >= 0. are different
        if (d >= 0. && !m_double_sided)
            return ClosestPointRecord{ false };
        d = abs(d);
        if (m_prevent_fd_artifacts)
            d *= 0.5; // avoid the artifact at the centerline
        for (int i = 0; i < m_nsteps; ++i) {
            if (d < m_epsilon) {
                // in shell
                auto its    = scene.closestPoint(p);
                its.contrib = contrib;
                return its;
            }

            if (scene.has_source()) {
                // interior term
                _float weight = 1.0;
                _float pdf    = 0.0;
                Vector dy;

#if 0
                if (m_use_IS_for_greens) {
                    // importance sampling                    
                    dy = sample_G_2D(sampler, d, pdf);
                    weight = dy.norm(); // Jacobian term from cartesian to polar
                } else {
                    // uniform sampling
                    dy = sample_uniform_inside_ball(sampler) * d;
                    pdf = 1.f / (M_PI * d * d);
                }
#else
                dy  = sample_uniform_inside_ball(sampler) * d;
                pdf = 1.f / (M_PI * d * d);
#endif

                _float G = eval_G_2D(dy, d);
                _float f = scene.source_function(p + dy);
                contrib += (pdf > 1e-5f ? f * G * weight / pdf : 0.0);
            }

            p = p + sample_uniform(sampler) * d;
            d = abs(scene.sdf(p));
        }
        return ClosestPointRecord{ false };
    }

    int    m_nwalks;
    int    m_nsteps;
    _float m_epsilon;
    bool   m_double_sided;
    bool   m_prevent_fd_artifacts;
    bool   m_use_IS_for_greens;
};

template <int N>
struct WoS {
    using Scene                = Scene<N>;
    using drVector             = dr::Vector<N>;
    using WoSDevice            = WoSDevice<N>;
    using drClosestPointRecord = dr::ClosestPointRecord<N>;
    static WoS create(int nwalks, int nsteps, _float epsilon, bool double_sided, bool prevent_fd_artifacts, bool use_IS_for_greens) {
        return WoS{
            nwalks,
            nsteps,
            epsilon,
            double_sided,
            prevent_fd_artifacts,
            use_IS_for_greens
        };
    }

    WoSDevice device() {
        return WoSDevice{
            m_nwalks,
            m_nsteps,
            m_epsilon,
            m_double_sided,
            m_prevent_fd_artifacts,
            m_use_IS_for_greens
        };
    }

    //! single walk
    drClosestPointRecord singleWalk(const drVector &_p, Scene &scene, dr::PCG32 &sampler) {
        int      n_points = drjit::width(_p);
        drVector p(_p);
        drjit::eval(p, sampler);
        cudaStream_t stream       = (cudaStream_t) jit_cuda_stream();
        SceneDevice  scene_device = scene.device();
        WoSDevice    wos_device   = device();
        //! AoS -> drjit SoA
        dr::Bool    cRec_valid   = drjit::zeros<dr::Bool>(n_points);
        drVector    cRec_p       = drjit::zeros<drVector>(n_points);
        drVector    cRec_n       = drjit::zeros<drVector>(n_points);
        dr::Float   cRec_t       = drjit::zeros<dr::Float>(n_points);   // only for 2d
        dr::Vector2 cRec_uv      = drjit::zeros<dr::Vector2>(n_points); // only for 3d
        dr::Int     cRec_prim_id = drjit::zeros<dr::Int>(n_points);
        dr::Float   cRec_contrib = drjit::zeros<dr::Float>(n_points);
        //! important. otherwise, they will point to the same memory.
        drjit::make_opaque(cRec_valid);
        drjit::make_opaque(cRec_p);
        drjit::make_opaque(cRec_n);
        drjit::make_opaque(cRec_t);
        drjit::make_opaque(cRec_uv);
        drjit::make_opaque(cRec_prim_id);
        drjit::make_opaque(cRec_contrib);

        if constexpr (N == 2) {
            auto input_begin  = thrust::make_zip_iterator(thrust::make_tuple(p.x().begin(), p.y().begin(),
                                                                             sampler.state.begin(), sampler.inc.begin()));
            auto output_begin = thrust::make_zip_iterator(thrust::make_tuple(cRec_valid.begin(),
                                                                             cRec_p.x().begin(), cRec_p.y().begin(),
                                                                             cRec_n.x().begin(), cRec_n.y().begin(),
                                                                             cRec_t.begin(), cRec_prim_id.begin(),
                                                                             cRec_contrib.begin()));
            auto begin        = thrust::make_zip_iterator(thrust::make_tuple(input_begin, output_begin));
            thrust::for_each(thrust::device.on((cudaStream_t) stream),
                             begin,
                             begin + (size_t) n_points,
                             [wos_device, scene_device] __device__(
                                 auto t) {
                                 auto &input        = thrust::get<0>(t);
                                 auto &p_x          = thrust::get<0>(input);
                                 auto &p_y          = thrust::get<1>(input);
                                 auto &s            = thrust::get<2>(input);
                                 auto &inc          = thrust::get<3>(input);
                                 auto &output       = thrust::get<1>(t);
                                 auto &v            = thrust::get<0>(output);
                                 auto &cRec_p_x     = thrust::get<1>(output);
                                 auto &cRec_p_y     = thrust::get<2>(output);
                                 auto &cRec_n_x     = thrust::get<3>(output);
                                 auto &cRec_n_y     = thrust::get<4>(output);
                                 auto &cRec_t       = thrust::get<5>(output);
                                 auto &id           = thrust::get<6>(output);
                                 auto &cRec_contrib = thrust::get<7>(output);

                                 auto rng     = Sampler::create(s, inc);
                                 auto its     = wos_device.walk(Vector2(p_x, p_y), scene_device, rng);
                                 v            = its.valid;
                                 cRec_p_x     = its.p.x();
                                 cRec_p_y     = its.p.y();
                                 cRec_n_x     = its.n.x();
                                 cRec_n_y     = its.n.y();
                                 cRec_t       = its.t;
                                 id           = its.prim_id;
                                 cRec_contrib = its.contrib;
                                 // step drjit sampler
                                 s   = rng.state;
                                 inc = rng.inc;
                             });

            return drClosestPointRecord{ cRec_valid,
                                         cRec_p,
                                         cRec_n,
                                         cRec_t,
                                         cRec_prim_id,
                                         cRec_contrib };
        } else if constexpr (N == 3) {
            auto input_begin   = thrust::make_zip_iterator(thrust::make_tuple(p.x().begin(), p.y().begin(), p.z().begin(),
                                                                              sampler.state.begin(), sampler.inc.begin()));
            auto output_begin  = thrust::make_zip_iterator(thrust::make_tuple(cRec_valid.begin(),
                                                                              cRec_p.x().begin(), cRec_p.y().begin(), cRec_p.z().begin(),
                                                                              cRec_n.x().begin(), cRec_n.y().begin(), cRec_n.z().begin(),
                                                                              cRec_uv.x().begin(), cRec_uv.y().begin(),
                                                                              cRec_prim_id.begin()));
            auto output_begin2 = thrust::make_zip_iterator(thrust::make_tuple(cRec_contrib.begin()));
            auto begin         = thrust::make_zip_iterator(thrust::make_tuple(input_begin, output_begin, output_begin2));
            thrust::for_each(thrust::device.on((cudaStream_t) stream),
                             begin,
                             begin + (size_t) n_points,
                             [wos_device, scene_device] __device__(
                                 auto t) {
                                 auto &input        = thrust::get<0>(t);
                                 auto &p_x          = thrust::get<0>(input);
                                 auto &p_y          = thrust::get<1>(input);
                                 auto &p_z          = thrust::get<2>(input);
                                 auto &s            = thrust::get<3>(input);
                                 auto &inc          = thrust::get<4>(input);
                                 auto &output       = thrust::get<1>(t);
                                 auto &v            = thrust::get<0>(output);
                                 auto &cRec_p_x     = thrust::get<1>(output);
                                 auto &cRec_p_y     = thrust::get<2>(output);
                                 auto &cRec_p_z     = thrust::get<3>(output);
                                 auto &cRec_n_x     = thrust::get<4>(output);
                                 auto &cRec_n_y     = thrust::get<5>(output);
                                 auto &cRec_n_z     = thrust::get<6>(output);
                                 auto &cRec_uv_x    = thrust::get<7>(output);
                                 auto &cRec_uv_y    = thrust::get<8>(output);
                                 auto &id           = thrust::get<9>(output);
                                 auto &output2      = thrust::get<2>(t);
                                 auto &cRec_contrib = thrust::get<0>(output2);

                                 auto rng     = Sampler::create(s, inc);
                                 auto its     = wos_device.walk(Vector3(p_x, p_y, p_z), scene_device, rng);
                                 v            = its.valid;
                                 cRec_p_x     = its.p.x();
                                 cRec_p_y     = its.p.y();
                                 cRec_p_z     = its.p.z();
                                 cRec_n_x     = its.n.x();
                                 cRec_n_y     = its.n.y();
                                 cRec_n_z     = its.n.z();
                                 cRec_uv_x    = its.uv.x();
                                 cRec_uv_y    = its.uv.y();
                                 id           = its.prim_id;
                                 cRec_contrib = its.contrib;
                                 // step drjit sampler
                                 s   = rng.state;
                                 inc = rng.inc;
                             });

            return drClosestPointRecord{ cRec_valid,
                                         cRec_p,
                                         cRec_n,
                                         cRec_uv,
                                         cRec_prim_id,
                                         cRec_contrib };
        }
    }

    dr::Float solve(const drVector &_p, Scene &scene, uint64_t seed) {
        int      n_points  = drjit::width(_p);
        int      n_repeats = min(100, m_nwalks);
        drVector p(_p);
        size_t   nsamples = drjit::width(p) * n_repeats;
        dr::Int  idx      = drjit::arange<dr::Int>(nsamples);
        idx /= n_repeats;
        p = drjit::gather<drVector>(p, idx);
        drjit::eval(p);
        size_t       n      = drjit::width(p);
        cudaStream_t stream = (cudaStream_t) jit_cuda_stream();
        //! SoA -> AoS
        SceneDevice                    scene_device = scene.device();
        thrust::device_vector<Sampler> samplers(n);
        auto                           samplers_ptr = samplers.data().get();
        //! seed samplers
        thrust::for_each(thrust::device.on((cudaStream_t) stream),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(n),
                         [seed, samplers_ptr] __device__(size_t i) {
                             samplers_ptr[i].seed(i, seed);
                         });
        dr::Float result = drjit::zeros<dr::Float>(n);
        drjit::make_opaque(result);
        auto      p_device   = to_device_vector(p);
        WoSDevice wos_device = device();
        //! WoS
        thrust::transform(thrust::device.on((cudaStream_t) stream),
                          thrust::make_zip_iterator(thrust::make_tuple(p_device.begin(), samplers.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(p_device.end(), samplers.end())),
                          result.begin(),
                          [wos_device, scene_device, n_repeats] __device__(auto t) {
                              auto   p      = thrust::get<0>(t);
                              auto   s      = thrust::get<1>(t);
                              _float result = 0.;
                              int    nwalks = wos_device.m_nwalks / n_repeats;
                              for (int i = 0; i < nwalks; ++i)
                                  result += wos_device.u(p, scene_device, s);
                              return result / nwalks;
                          });

        dr::Float values = drjit::zeros<dr::Float>(n_points);
        drjit::scatter_reduce(ReduceOp::Add, values, result, idx);
        return values / n_repeats;
    }

    int    m_nwalks;
    int    m_nsteps;
    _float m_epsilon;
    bool   m_double_sided;
    bool   m_prevent_fd_artifacts;
    bool   m_use_IS_for_greens;
};

template <>
inline dr::ClosestPointRecord<2> WoS<2>::singleWalk(const drVector &_p, Scene &scene, dr::PCG32 &sampler) {
    int      n_points = drjit::width(_p);
    drVector p(_p);
    drjit::eval(p, sampler);
    cudaStream_t stream       = (cudaStream_t) jit_cuda_stream();
    SceneDevice  scene_device = scene.device();
    WoSDevice    wos_device   = device();
    //! AoS -> drjit SoA
    dr::Bool    cRec_valid   = drjit::zeros<dr::Bool>(n_points);
    drVector    cRec_p       = drjit::zeros<drVector>(n_points);
    drVector    cRec_n       = drjit::zeros<drVector>(n_points);
    dr::Float   cRec_t       = drjit::zeros<dr::Float>(n_points);   // only for 2d
    dr::Vector2 cRec_uv      = drjit::zeros<dr::Vector2>(n_points); // only for 3d
    dr::Int     cRec_prim_id = drjit::zeros<dr::Int>(n_points);
    dr::Float   cRec_contrib = drjit::zeros<dr::Float>(n_points);
    //! important. otherwise, they will point to the same memory.
    drjit::make_opaque(cRec_valid);
    drjit::make_opaque(cRec_p);
    drjit::make_opaque(cRec_n);
    drjit::make_opaque(cRec_t);
    drjit::make_opaque(cRec_uv);
    drjit::make_opaque(cRec_prim_id);
    drjit::make_opaque(cRec_contrib);

    auto input_begin  = thrust::make_zip_iterator(thrust::make_tuple(p.x().begin(), p.y().begin(),
                                                                     sampler.state.begin(), sampler.inc.begin()));
    auto output_begin = thrust::make_zip_iterator(thrust::make_tuple(cRec_valid.begin(),
                                                                     cRec_p.x().begin(), cRec_p.y().begin(),
                                                                     cRec_n.x().begin(), cRec_n.y().begin(),
                                                                     cRec_t.begin(), cRec_prim_id.begin(),
                                                                     cRec_contrib.begin()));
    auto begin        = thrust::make_zip_iterator(thrust::make_tuple(input_begin, output_begin));
    thrust::for_each(thrust::device.on((cudaStream_t) stream),
                     begin,
                     begin + (size_t) n_points,
                     [wos_device, scene_device] __device__(
                         auto t) {
                         auto &input        = thrust::get<0>(t);
                         auto &p_x          = thrust::get<0>(input);
                         auto &p_y          = thrust::get<1>(input);
                         auto &s            = thrust::get<2>(input);
                         auto &inc          = thrust::get<3>(input);
                         auto &output       = thrust::get<1>(t);
                         auto &v            = thrust::get<0>(output);
                         auto &cRec_p_x     = thrust::get<1>(output);
                         auto &cRec_p_y     = thrust::get<2>(output);
                         auto &cRec_n_x     = thrust::get<3>(output);
                         auto &cRec_n_y     = thrust::get<4>(output);
                         auto &cRec_t       = thrust::get<5>(output);
                         auto &id           = thrust::get<6>(output);
                         auto &cRec_contrib = thrust::get<7>(output);

                         auto rng     = Sampler::create(s, inc);
                         auto its     = wos_device.walk(Vector2(p_x, p_y), scene_device, rng);
                         v            = its.valid;
                         cRec_p_x     = its.p.x();
                         cRec_p_y     = its.p.y();
                         cRec_n_x     = its.n.x();
                         cRec_n_y     = its.n.y();
                         cRec_t       = its.t;
                         id           = its.prim_id;
                         cRec_contrib = its.contrib;
                         // step drjit sampler
                         s   = rng.state;
                         inc = rng.inc;
                     });

    return drClosestPointRecord{ cRec_valid,
                                 cRec_p,
                                 cRec_n,
                                 cRec_t,
                                 cRec_prim_id,
                                 cRec_contrib };
}

template <>
inline dr::ClosestPointRecord<3> WoS<3>::singleWalk(const drVector &_p, Scene &scene, dr::PCG32 &sampler) {
    int      n_points = drjit::width(_p);
    drVector p(_p);
    drjit::eval(p, sampler);
    cudaStream_t stream       = (cudaStream_t) jit_cuda_stream();
    SceneDevice  scene_device = scene.device();
    WoSDevice    wos_device   = device();
    //! AoS -> drjit SoA
    dr::Bool    cRec_valid   = drjit::zeros<dr::Bool>(n_points);
    drVector    cRec_p       = drjit::zeros<drVector>(n_points);
    drVector    cRec_n       = drjit::zeros<drVector>(n_points);
    dr::Float   cRec_t       = drjit::zeros<dr::Float>(n_points);   // only for 2d
    dr::Vector2 cRec_uv      = drjit::zeros<dr::Vector2>(n_points); // only for 3d
    dr::Int     cRec_prim_id = drjit::zeros<dr::Int>(n_points);
    dr::Float   cRec_contrib = drjit::zeros<dr::Float>(n_points);
    //! important. otherwise, they will point to the same memory.
    drjit::make_opaque(cRec_valid);
    drjit::make_opaque(cRec_p);
    drjit::make_opaque(cRec_n);
    drjit::make_opaque(cRec_t);
    drjit::make_opaque(cRec_uv);
    drjit::make_opaque(cRec_prim_id);
    drjit::make_opaque(cRec_contrib);

    auto input_begin   = thrust::make_zip_iterator(thrust::make_tuple(p.x().begin(), p.y().begin(), p.z().begin(),
                                                                      sampler.state.begin(), sampler.inc.begin()));
    auto output_begin  = thrust::make_zip_iterator(thrust::make_tuple(cRec_valid.begin(),
                                                                      cRec_p.x().begin(), cRec_p.y().begin(), cRec_p.z().begin(),
                                                                      cRec_n.x().begin(), cRec_n.y().begin(), cRec_n.z().begin(),
                                                                      cRec_uv.x().begin(), cRec_uv.y().begin(),
                                                                      cRec_prim_id.begin()));
    auto output_begin2 = thrust::make_zip_iterator(thrust::make_tuple(cRec_contrib.begin()));
    auto begin         = thrust::make_zip_iterator(thrust::make_tuple(input_begin, output_begin, output_begin2));
    thrust::for_each(thrust::device.on((cudaStream_t) stream),
                     begin,
                     begin + (size_t) n_points,
                     [wos_device, scene_device] __device__(
                         auto t) {
                         auto &input        = thrust::get<0>(t);
                         auto &p_x          = thrust::get<0>(input);
                         auto &p_y          = thrust::get<1>(input);
                         auto &p_z          = thrust::get<2>(input);
                         auto &s            = thrust::get<3>(input);
                         auto &inc          = thrust::get<4>(input);
                         auto &output       = thrust::get<1>(t);
                         auto &v            = thrust::get<0>(output);
                         auto &cRec_p_x     = thrust::get<1>(output);
                         auto &cRec_p_y     = thrust::get<2>(output);
                         auto &cRec_p_z     = thrust::get<3>(output);
                         auto &cRec_n_x     = thrust::get<4>(output);
                         auto &cRec_n_y     = thrust::get<5>(output);
                         auto &cRec_n_z     = thrust::get<6>(output);
                         auto &cRec_uv_x    = thrust::get<7>(output);
                         auto &cRec_uv_y    = thrust::get<8>(output);
                         auto &id           = thrust::get<9>(output);
                         auto &output2      = thrust::get<2>(t);
                         auto &cRec_contrib = thrust::get<0>(output2);

                         auto rng     = Sampler::create(s, inc);
                         auto its     = wos_device.walk(Vector3(p_x, p_y, p_z), scene_device, rng);
                         v            = its.valid;
                         cRec_p_x     = its.p.x();
                         cRec_p_y     = its.p.y();
                         cRec_p_z     = its.p.z();
                         cRec_n_x     = its.n.x();
                         cRec_n_y     = its.n.y();
                         cRec_n_z     = its.n.z();
                         cRec_uv_x    = its.uv.x();
                         cRec_uv_y    = its.uv.y();
                         id           = its.prim_id;
                         cRec_contrib = its.contrib;
                         // step drjit sampler
                         s   = rng.state;
                         inc = rng.inc;
                     });

    return drClosestPointRecord{ cRec_valid,
                                 cRec_p,
                                 cRec_n,
                                 cRec_uv,
                                 cRec_prim_id,
                                 cRec_contrib };
}

}; // namespace wos_cuda