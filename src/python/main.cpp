#include <binding.h>

#define MODULE_NAME wos_ext

PY_DECLARE(Scene);
PY_DECLARE(WoS);
PY_DECLARE(Test);

PYBIND11_MODULE(MODULE_NAME, m) {
    py::module::import("drjit");
    py::module::import("drjit.cuda");
    py::module::import("drjit.cuda.ad");
    m.attr("__version__") = "0.0.1";
    m.attr("__name__")    = "wos";
    PY_IMPORT(Scene);
    PY_IMPORT(WoS);
    PY_IMPORT(Test);
}