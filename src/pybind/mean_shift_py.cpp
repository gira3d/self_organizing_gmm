#include <self_organizing_gmm/MeanShift2D.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(mean_shift_py, m)
{
  py::class_<MeanShift2D>(m, "MeanShift")
      .def(py::init())
      .def(py::init<float>())
      .def(py::init<float, bool, std::string, std::string>())
      .def("get_num_modes", &MeanShift2D::get_num_modes)
      .def("get_mode_centers", &MeanShift2D::get_mode_centers)
      .def("fit", &MeanShift2D::fit);
}
