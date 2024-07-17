#include <fwd.h>
#include <binding.h>

class Test {
public:
    Test() {}
    Test(const dr::Vector2 &data) : data(data) {}
    dr::Vector2 data;
};

PY_EXPORT(Test) {
    py::class_<Test>(m, "Test")
        .def(py::init<>())
        .def(py::init<const dr::Vector2 &>())
        .def_readwrite("data", &Test::data);
}