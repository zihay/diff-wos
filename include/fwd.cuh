#pragma once

#include <fwd.h>
#include <thrust/async/copy.h>
#include <thrust/async/transform.h>
#include <thrust/detail/config.h>
#include <thrust/device_vector.h>

#include <Eigen/Dense>

namespace wos_cuda {
template <int DIM>
using Vector  = Eigen::Matrix<_float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <int DIM>
using Vectori  = Eigen::Matrix<int, DIM, 1>;
using Vector2i = Vectori<2>;
using Vector3i = Vectori<3>;

template <int DIM>
using Array  = Eigen::Array<_float, DIM, 1>;
using Array2 = Array<2>;
using Array3 = Array<3>;

inline thrust::device_vector<_float> to_device_vector(const dr::Float &vec) {
    thrust::device_vector<_float> result(drjit::width(vec));
    auto                          begin  = vec.begin();
    auto                          end    = vec.end();
    cudaStream_t                  stream = (cudaStream_t) jit_cuda_stream();
    thrust::transform(thrust::device.on(stream),
                      begin, end, result.begin(),
                      [] __device__(_float t) {
                          return t;
                      });
    return result;
}

inline thrust::device_vector<Vector2> to_device_vector(const dr::Vector2 &vec) {
    thrust::device_vector<Vector2> result(drjit::width(vec));
    auto                           begin  = thrust::make_zip_iterator(thrust::make_tuple(vec[0].begin(), vec[1].begin()));
    auto                           end    = thrust::make_zip_iterator(thrust::make_tuple(vec[0].end(), vec[1].end()));
    cudaStream_t                   stream = (cudaStream_t) jit_cuda_stream();
    thrust::transform(thrust::device.on(stream),
                      begin, end, result.begin(),
                      [] __device__(thrust::tuple<_float, _float> t) {
                          return Vector2(thrust::get<0>(t), thrust::get<1>(t));
                      });
    return result;
}

inline thrust::device_vector<Vector3> to_device_vector(const dr::Vector3 &vec) {
    thrust::device_vector<Vector3> result(drjit::width(vec));
    auto                           begin  = thrust::make_zip_iterator(thrust::make_tuple(vec[0].begin(), vec[1].begin(), vec[2].begin()));
    auto                           end    = thrust::make_zip_iterator(thrust::make_tuple(vec[0].end(), vec[1].end(), vec[2].end()));
    cudaStream_t                   stream = (cudaStream_t) jit_cuda_stream();
    thrust::transform(thrust::device.on(stream),
                      begin, end, result.begin(),
                      [] __device__(thrust::tuple<_float, _float, _float> t) {
                          return Vector3(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
                      });
    return result;
}

inline thrust::device_vector<Vector2i> to_device_vector(const dr::Vector2i &vec) {
    thrust::device_vector<Vector2i> result(drjit::width(vec));
    auto                            begin  = thrust::make_zip_iterator(thrust::make_tuple(vec[0].begin(), vec[1].begin()));
    auto                            end    = thrust::make_zip_iterator(thrust::make_tuple(vec[0].end(), vec[1].end()));
    cudaStream_t                    stream = (cudaStream_t) jit_cuda_stream();
    thrust::transform(thrust::device.on(stream),
                      begin, end, result.begin(),
                      [] __device__(thrust::tuple<int, int> t) {
                          return Vector2i(thrust::get<0>(t), thrust::get<1>(t));
                      });
    return result;
}

inline thrust::device_vector<Vector3i> to_device_vector(const dr::Vector3i &vec) {
    thrust::device_vector<Vector3i> result(drjit::width(vec));
    auto                            begin  = thrust::make_zip_iterator(thrust::make_tuple(vec[0].begin(), vec[1].begin(), vec[2].begin()));
    auto                            end    = thrust::make_zip_iterator(thrust::make_tuple(vec[0].end(), vec[1].end(), vec[2].end()));
    cudaStream_t                    stream = (cudaStream_t) jit_cuda_stream();
    thrust::transform(thrust::device.on(stream),
                      begin, end, result.begin(),
                      [] __device__(thrust::tuple<int, int, int> t) {
                          return Vector3i(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
                      });
    return result;
}

inline dr::Vector2 to_drjit_vector(const thrust::device_vector<Vector2> &vec) {
    dr::Vector2 result = drjit::zeros<dr::Vector2>(vec.size());
    drjit::make_opaque(result);
    auto  begin  = thrust::make_zip_iterator(thrust::make_tuple(vec.begin(), result[0].begin(), result[1].begin()));
    auto  end    = thrust::make_zip_iterator(thrust::make_tuple(vec.end(), result[0].end(), result[1].end()));
    void *stream = jit_cuda_stream();
    thrust::for_each(thrust::device.on((cudaStream_t) stream), begin, end,
                     [] __device__(thrust::tuple<Vector2, _float &, _float &> t) {
                         thrust::get<1>(t) = thrust::get<0>(t)[0];
                         thrust::get<2>(t) = thrust::get<0>(t)[1];
                     });
    return result;
}
}; // namespace wos_cuda

template <typename T>
__host__ __device__ inline T lerp(T a, T b, _float t) {
    return a + t * (b - a);
}

template <typename T>
__host__ __device__ inline T interpolate(T a, T b, T c, const wos_cuda::Vector2 &uv) {
    b *uv.x() + c *uv.y() + a * (1. - uv.x() - uv.y());
}

template <typename T>
__device__ T clamp(T x, T min, T max) {
    return thrust::max(thrust::min(x, max), min);
}

template <typename T>
__device__ T sign(T x) {
    if (x > 0)
        return 1.0;
    else
        return -1.0;
}

template <typename T>
__host__ __device__ inline T max(T a, T b, T c) {
    return max(max(a, b), c);
}

template <typename T>
__host__ __device__ inline T min(T a, T b, T c) {
    return min(min(a, b), c);
}