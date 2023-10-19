#include <self_organizing_gmm/KInit.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/// @brief Function to generate bindings over the KInit class for different
/// template parameters
/// @author Kshitij Goel
/// @tparam T Datatype (e.g., float, double)
/// @tparam D Dimensions of the data (e.g., 1, 2, 3, 4)
/// @param m Pybind module where the bindings will be added.
/// @param typestr Additional string appended to the class name.
template <typename T, uint32_t D>
void binding_generator(py::module &m, std::string &typestr)
{
  using KInitClass = sogmm::KInit<T, D>;
  std::string pyclass_name = std::string("KInit") + typestr;
  py::class_<KInitClass>(m, pyclass_name.c_str(), py::dynamic_attr())
      .def(py::init())
      .def("cumsum", &KInitClass::cumSum)
      .def("search_sorted", &KInitClass::searchSorted)
      .def("euclidean_dists", &KInitClass::euclideanDistancesSq)
      .def("fit", &KInitClass::fit)
      .def("resp_calc", &KInitClass::respCalc)
      .def("get_resp_mat", &KInitClass::getRespMat);
}

PYBIND11_MODULE(kinit_py, g)
{
  std::string t1 = "f1CPU";
  binding_generator<float, 1>(g, t1);

  std::string t2 = "f2CPU";
  binding_generator<float, 2>(g, t2);

  std::string t3 = "f3CPU";
  binding_generator<float, 3>(g, t3);

  std::string t4 = "f4CPU";
  binding_generator<float, 4>(g, t4);
}