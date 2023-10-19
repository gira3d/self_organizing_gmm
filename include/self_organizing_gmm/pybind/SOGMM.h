#include <self_organizing_gmm/SOGMMLearner.h>
#include <self_organizing_gmm/SOGMMInference.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace sogmm
{
  namespace cpu
  {
    template <typename T, uint32_t D>
    void container_binding_generator(py::module &m, std::string &typestr)
    {
      using Container = sogmm::cpu::SOGMM<T, D>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      py::class_<Container>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                            py::dynamic_attr())
          .def(py::init(), "Default empty constructor.")
          .def(py::init<const Container &>(),
               "Copy from an existing container.",
               py::arg("that"))
          .def(py::init<const uint32_t &>(),
               "Initialize zero members for the given number of components.",
               py::arg("n_components"))
          .def(py::init<const typename Container::Vector &,
                        const typename Container::MatrixXD &,
                        const typename Container::MatrixXC &,
                        const uint32_t &>(),
               py::arg("weights"),
               py::arg("means"),
               py::arg("covariances"),
               py::arg("support_size"))
          .def_readwrite("n_components_", &Container::n_components_,
                         "Number of components in this GMM.")
          .def_readwrite("support_size_", &Container::support_size_,
                         "Number of points in the support of this GMM.")
          .def_readwrite("weights_", &Container::weights_, "All weights.")
          .def_readwrite("means_", &Container::means_, "All means.")
          .def_readwrite("covariances_", &Container::covariances_,
                         "All covariances.")
          .def_readwrite("precisions_cholesky_", &Container::precisions_cholesky_,
                         "All cholesky decompositions of the precision matrices.")
          .def_readwrite("covariances_cholesky_", &Container::covariances_cholesky_,
                         "All cholesky decompositions of the covariance matrices.")
          .def("normalize_weights", &Container::normalizeWeights,
               "Normalize the weight vector.")
          .def("update_cholesky", &Container::updateCholesky,
               "Update covariances_cholesky_ and precisions_cholesky_.")
          .def("merge", &Container::merge,
               "Merge another container into this container.")
          .def("submap_from_indices", &Container::submapFromIndices,
               "Create a sub GMM from supplied list of indices.")
          .def(py::pickle(
                   [](const Container &g)
                   {
                     return py::make_tuple(g.weights_, g.means_, g.covariances_, g.support_size_);
                   },
                   [](py::tuple t)
                   {
                     Container g = Container(t[0].cast<typename Container::Vector>(),
                                             t[1].cast<typename Container::MatrixXD>(),
                                             t[2].cast<typename Container::MatrixXC>(),
                                             t[3].cast<uint32_t>());
                     return g;
                   }),
               "Serialization/Deserialization through pickling.");
    }

    template <typename T>
    void learner_binding_generator(py::module &m, std::string &typestr)
    {
      using Learner = sogmm::cpu::SOGMMLearner<T>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      py::class_<Learner>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                          py::dynamic_attr())
          .def(py::init(), "Default empty constructor.")
          .def(py::init<const float &>(), "Initialize using the bandwidth parameter")
          .def("fit", &Learner::fit)
          .def("fit_em", &Learner::fit_em);
    }

    template <typename T>
    void inference_binding_generator(py::module &m, std::string &typestr)
    {
      using Container = sogmm::cpu::SOGMM<T, 4>;
      using Inference = sogmm::cpu::SOGMMInference<T>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      py::class_<Inference>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                            py::dynamic_attr())
          .def(py::init(), "Default empty constructor.")
          .def("generate_pcld_4d", &Inference::generatePointCloud4D)
          .def("generate_pcld_3d", &Inference::generatePointCloud3D)
          .def("reconstruct", &Inference::reconstruct)
          .def("color_query", &Inference::colorQuery)
          .def("reconstruct_fast", &Inference::reconstructFast)
          .def("score_4d", &Inference::score4D)
          .def("score_3d", &Inference::score3D);
    }
  }
}