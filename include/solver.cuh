#pragma once
#include <fwd.cuh>
#include <scene.cuh>

namespace wos_cuda {
struct HitRecord {
};

struct Solver {
    using SceneDevice = SceneDevice<2>;
    __device__ virtual _float solve(const Vector2 &p, const SceneDevice &scene, uint64_t seed) {
        return 0.;
    }
};
}; // namespace wos_cuda