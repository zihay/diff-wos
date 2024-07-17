#pragma once
#include <curand_kernel.h>

#include <fwd.cuh>
namespace wos_cuda {
struct Sampler {
    __device__ void seed(size_t idx, uint64_t seed_value) {
        curand_init(seed_value, idx, 0, &m_curand_state);
    }

    __device__ _float next_1d() {
        return curand_uniform(&m_curand_state);
    }

    __device__ Vector2 next_2d() {
        return Vector2(next_1d(), next_1d());
    }

    __device__ Vector3 next_3d() {
        return Vector3(next_1d(), next_1d(), next_1d());
    }

    curandState_t m_curand_state;
};
} // namespace wos_cuda