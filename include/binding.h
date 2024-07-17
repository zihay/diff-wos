#pragma once
#include <fwd.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define PY_DECLARE(Name)          extern void python_export_##Name(py::module_ &m)
#define PY_EXPORT(Name)           void python_export_##Name(py::module_ &m)
#define PY_IMPORT(Name)           python_export_##Name(m)
#define PY_IMPORT_SUBMODULE(Name) python_export_##Name(Name)

#define EI_PY_IMPORT_TYPES(...)       using T           = EI_VARIANT_T; \
                                     constexpr int DIM = EI_VARIANT_DIM;
