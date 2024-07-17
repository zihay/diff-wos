#include <binding.h>

#include <wos.cuh>

template <int N>
void export_wos(py::module_ &m, const std::string &name = "WoS") {
    using WoS = wos_cuda::WoS<N>;
    py::class_<WoS>(m, name.c_str())
        .def(py::init(&WoS::create),
             py::arg("nwalks"), py::arg("nsteps") = 32, py::arg("epsilon") = 1e-3, 
             py::arg("double_sided") = false, py::arg("prevent_fd_artifacts") = false,
             py::arg("use_IS_for_greens") = true)
        .def("solve", &WoS::solve)
        .def("single_walk", &WoS::singleWalk)
        .def_readwrite("nwalks", &WoS::m_nwalks)
        .def_readwrite("nsteps", &WoS::m_nsteps)
        .def_readwrite("epsilon", &WoS::m_epsilon)
        .def_readwrite("double_sided", &WoS::m_double_sided)
        .def_readwrite("use_IS_for_greens", &WoS::m_use_IS_for_greens);        
}

PY_EXPORT(WoS) {
    export_wos<2>(m);
    export_wos<3>(m, "WoS3D");
}