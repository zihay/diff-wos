#include <drjit/array.h>
#include <drjit/math.h>
#include <binding.h>

#include <scene.cuh>

template <int N>
void export_scene(py::module_ &m);

template <>
void export_scene<2>(py::module_ &m) {
    using AABB               = wos_cuda::AABB<2>;
    using Scene              = wos_cuda::Scene<2>;
    using closestPointRecord = dr::ClosestPointRecord<2>;
    py::class_<AABB>(m, "AABB")
        .def_readwrite("min", &AABB::min)
        .def_readwrite("max", &AABB::max);

    py::class_<closestPointRecord>(m, "ClosestPointRecord")
        .def(py::init(&closestPointRecord::create))
        .def_readwrite("valid", &closestPointRecord::valid)
        .def_readwrite("p", &closestPointRecord::p)
        .def_readwrite("n", &closestPointRecord::n)
        .def_readwrite("t", &closestPointRecord::t)
        .def_readwrite("prim_id", &closestPointRecord::prim_id)
        .def_readwrite("contrib", &closestPointRecord::contrib);

    py::class_<Scene>(m, "Scene")
        .def(py::init<>())
        .def(py::init<const dr::Vector2 &, const dr::Vector2i &, const dr::Float &, bool>(),
             py::arg("vertices"), py::arg("indices"), py::arg("values"), py::arg("use_bvh") = false)
        .def(py::init<const dr::Vector2 &, const dr::Vector2i &, const dr::Float &, bool, int, const dr::Float &>(),
             py::arg("vertices"), py::arg("indices"), py::arg("values"), py::arg("use_bvh"), py::arg("source_type"), py::arg("source_params"))
        .def("from_vertices", &Scene::fromVertices)
        .def_readwrite("vertices", &Scene::m_vertices)
        .def_readwrite("indices", &Scene::m_indices)
        .def_readwrite("normals", &Scene::m_normals)
        .def_readwrite("values", &Scene::m_values)
        .def_readwrite("aabb", &Scene::m_aabb)
        .def_readwrite("use_bvh", &Scene::m_use_bvh)
        .def("largest_inscribed_ball", &Scene::largestInscribedBall,
             py::arg("its"), py::arg("epsilon") = 1e-3)
        .def("closest_point_preliminary", &Scene::closestPointPreliminary)
        .def("closest_point", &Scene::closestPoint)
        .def("dirichlet", &Scene::dirichlet)
        .def_readwrite("source_type", &Scene::m_source_type)
        .def_readwrite("source_params", &Scene::m_source_params);
}

template <>
void export_scene<3>(py::module_ &m) {
    using AABB               = wos_cuda::AABB<3>;
    using Scene              = wos_cuda::Scene<3>;
    using closestPointRecord = dr::ClosestPointRecord<3>;
    py::class_<AABB>(m, "AABB3D")
        .def_readwrite("min", &AABB::min)
        .def_readwrite("max", &AABB::max);

    py::class_<closestPointRecord>(m, "ClosestPointRecord3D")
        .def(py::init(&closestPointRecord::create))
        .def_readwrite("valid", &closestPointRecord::valid)
        .def_readwrite("p", &closestPointRecord::p)
        .def_readwrite("n", &closestPointRecord::n)
        .def_readwrite("uv", &closestPointRecord::uv)
        .def_readwrite("prim_id", &closestPointRecord::prim_id)
        .def_readwrite("contrib", &closestPointRecord::contrib);

    py::class_<Scene>(m, "Scene3D")
        .def(py::init<>())
        .def(py::init<const dr::Vector3 &, const dr::Vector3i &, const dr::Float &, bool>(),
             py::arg("vertices"), py::arg("indices"), py::arg("values"), py::arg("use_bvh") = false)
        .def("from_vertices", &Scene::fromVertices)
        .def_readwrite("vertices", &Scene::m_vertices)
        .def_readwrite("indices", &Scene::m_indices)
        .def_readwrite("normals", &Scene::m_normals)
        .def_readwrite("values", &Scene::m_values)
        .def_readwrite("aabb", &Scene::m_aabb)
        .def("largest_inscribed_ball", &Scene::largestInscribedBall,
             py::arg("its"), py::arg("epsilon") = 1e-3)
        .def("closest_point_preliminary", &Scene::closestPointPreliminary)
        .def("closest_point", &Scene::closestPoint)
        .def("dirichlet", &Scene::dirichlet);
}

PY_EXPORT(Scene) {
    export_scene<2>(m);
    export_scene<3>(m);
}