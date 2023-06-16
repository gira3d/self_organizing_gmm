#include <self_organizing_gmm/KInit.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T, uint32_t D>
void binding_generator(py::module &m, std::string &typestr)
{
  using KInitClass = KInit<T, D>;
  std::string pyclass_name = std::string("KInit") + typestr;
  py::class_<KInitClass>(m, pyclass_name.c_str(), py::dynamic_attr())
      .def(py::init())
      .def(py::init<unsigned int>())
      .def(py::init<unsigned int, bool, std::string, std::string>())
      .def_readwrite("n_components_", &KInitClass::n_components_)
      .def("cumsum", &KInitClass::cumsum)
      .def("search_sorted", &KInitClass::search_sorted)
      .def("euclidean_dists", &KInitClass::euclidean_distances_sq)
      .def("resp_calc", &KInitClass::resp_calc)
      .def(py::pickle(
          [](const KInitClass &g)
          {
            return py::make_tuple(
                g.n_components_);
          },
          [](py::tuple t)
          {
            KInitClass g;
            g.n_components_ = t[0].cast<unsigned int>();
            return g;
          }));
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