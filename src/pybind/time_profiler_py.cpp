#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <self_organizing_gmm/TimeProfiler.h>

namespace py = pybind11;

PYBIND11_MODULE(time_profiler_py, m)
{
  py::class_<TimeProfiler>(m, "TimeProfiler")
  .def(py::init())
  .def("tic", &TimeProfiler::tic)
  .def("toc", &TimeProfiler::toc);
}
