#pragma once
#include <curand_kernel.h>

#include <fwd.cuh>

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL
namespace wos_cuda {
// copy from drjit random.h
struct Sampler {
    //! use drjit sampler state to init
    static __device__ Sampler create(uint64_t state, uint64_t inc) {
        Sampler sampler;
        sampler.state = state;
        sampler.inc   = inc;
        return sampler;
    }

    __device__ void seed(size_t   index,
                         uint64_t initstate = PCG32_DEFAULT_STATE,
                         uint64_t initseq   = PCG32_DEFAULT_STREAM) {
        state = 0;
        inc   = ((initseq + index) << 1) | 1u;
        next_uint32();
        state += initstate;
        next_uint32();
    }

    __device__ uint32_t next_uint32() {
        uint64_t oldstate = state;
        //! fma yield wrong result
        state = oldstate * uint64_t(PCG32_MULT) + inc;
        uint32_t xorshift = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot      = uint32_t(oldstate >> 59u);
        return (xorshift >> rot) | (xorshift << ((-uint32_t(rot)) & 31));
    }

    __device__ _float next_1d() {
        auto v = ((next_uint32() >> 9) | 0x3f800000u);
        _float result;
        memcpy(&result, &v, sizeof(_float));
        return result - 1.;
    }

    __device__ Vector2 next_2d() {
        return Vector2(next_1d(), next_1d());
    }

    __device__ Vector3 next_3d() {
        return Vector3(next_1d(), next_1d(), next_1d());
    }

    uint64_t state;
    uint64_t inc;
};
} // namespace wos_cuda