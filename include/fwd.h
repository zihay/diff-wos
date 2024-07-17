#pragma once

#include <drjit/autodiff.h>
#include <drjit/jit.h>
#include <drjit/matrix.h>
#include <drjit/quaternion.h>
#include <drjit/random.h>
#include <drjit/struct.h>
#include <drjit/vcall.h>

using _float = float;
// namespace dr = drjit;
#ifndef M_PI
#define M_PI     3.14159265358979323846f
#endif
namespace dr {
template <typename T>
using CUDAArrayAD = drjit::DiffArray<drjit::CUDAArray<T>>;
using Bool        = CUDAArrayAD<bool>;
using Int         = CUDAArrayAD<int>;
using UInt        = CUDAArrayAD<unsigned int>;
using UInt64      = CUDAArrayAD<uint64_t>;
using Float       = CUDAArrayAD<_float>;
template <int DIM>
using Vector  = drjit::Array<Float, DIM>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <int DIM>
using Vectori  = drjit::Array<Int, DIM>;
using Vector2i = Vectori<2>;
using Vector3i = Vectori<3>;

template <int DIM>
using Array  = drjit::Array<Float, DIM>;
using Array2 = Array<2>;
using Array3 = Array<3>;

template <int DIM>
using Arrayi  = drjit::Array<Int, DIM>;
using Array2i = Arrayi<2>;
using Array3i = Arrayi<3>;

using PCG32 = drjit::PCG32<UInt64>;
} // namespace dr