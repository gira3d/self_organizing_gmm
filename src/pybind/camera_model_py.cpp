#include <self_organizing_gmm/CameraModel.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(camera_model_py, m)
{
  py::class_<CameraModel>(m, "CameraModel")
      .def(py::init())
      .def(py::init<Eigen::Matrix3f>())
      .def(py::init<Eigen::Matrix3f, size_t, size_t>())
      .def(py::init<float, float, float, float, size_t, size_t>())
      .def_readwrite("w", &CameraModel::im_w_)
      .def_readwrite("h", &CameraModel::im_h_)
      .def_readwrite("K", &CameraModel::intrinsic_matrix_)
      .def("to_3d", &CameraModel::to_3d)
      .def("to_2d", &CameraModel::to_2d)
      .def("to_2d_dim", &CameraModel::to_2d_dim);
}
