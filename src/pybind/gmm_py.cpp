#include <self_organizing_gmm/GMM.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T, uint32_t D>
void binding_generator(py::module& m, std::string& typestr)
{
  static constexpr uint32_t C = D * D;
  using GMMClass = GMM<T, D>;
  std::string pyclass_name = std::string("GMM") + typestr;
  py::class_<GMMClass>(m, pyclass_name.c_str(), py::dynamic_attr())
      .def(py::init())
      .def(py::init<unsigned int>())
      .def(py::init<unsigned int, T, T, unsigned int>())
      .def(py::init<unsigned int, bool, std::string, std::string>())
      .def(py::init<unsigned int, T, T, unsigned int, bool, std::string, std::string>())
      .def_readwrite("n_components_", &GMMClass::n_components_)
      .def_readwrite("tol_", &GMMClass::tol_)
      .def_readwrite("reg_covar_", &GMMClass::reg_covar_)
      .def_readwrite("max_iter_", &GMMClass::max_iter_)
      .def_readwrite("support_size_", &GMMClass::support_size_)
      .def_readwrite("weights_", &GMMClass::weights_)
      .def_readwrite("means_", &GMMClass::means_)
      .def_readwrite("covariances_", &GMMClass::covariances_)
      .def_readwrite("precisions_cholesky_", &GMMClass::precisions_cholesky_)
      .def_readwrite("resp_", &GMMClass::resp_)
      .def("compute_log_det_cholesky", &GMMClass::computeLogDetCholesky)
      .def("compute_precision_cholesky", &GMMClass::computeCholesky)
      .def("estimate_log_gaussian_prob", &GMMClass::estimateLogGaussianProb)
      .def("estimate_log_prob", &GMMClass::estimateLogProb)
      .def("log_sum_exp_cols", &GMMClass::logSumExpCols)
      .def("estimate_weighted_log_prob", &GMMClass::estimateWeightedLogProb)
      .def("estimate_log_prob_resp", &GMMClass::estimateLogProbResp)
      .def("estimate_gaussian_parameters",
           &GMMClass::estimateGaussianParameters)
      .def("e_step", &GMMClass::eStep)
      .def("m_step", &GMMClass::mStep)
      .def("fit", &GMMClass::fit)
      .def("sample", &GMMClass::sample)
      .def("score_samples", &GMMClass::scoreSamples)
      .def("score", &GMMClass::score)
      .def("merge", &GMMClass::merge)
      .def("color_conditional", &GMMClass::colorConditional)
      .def(py::pickle(
          [](const GMMClass& g) {
            return py::make_tuple(
                g.n_components_, g.tol_, g.reg_covar_, g.max_iter_,
                g.support_size_, g.weights_, g.means_, g.covariances_,
                g.precisions_cholesky_, g.covariances_cholesky_);
          },
          [](py::tuple t) {
            GMMClass g;
            g.n_components_ = t[0].cast<unsigned int>();
            g.tol_ = t[1].cast<T>();
            g.reg_covar_ = t[2].cast<T>();
            g.max_iter_ = t[3].cast<unsigned int>();
            g.support_size_ = t[4].cast<unsigned int>();
            g.weights_ = t[5].cast<Eigen::Matrix<T, Eigen::Dynamic, 1>>();
            g.means_ = t[6].cast<
                Eigen::Matrix<T, Eigen::Dynamic, D, (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>();
            g.covariances_ = t[7].cast<
                Eigen::Matrix<T, Eigen::Dynamic, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>();
            g.precisions_cholesky_ = t[8].cast<
                Eigen::Matrix<T, Eigen::Dynamic, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>();
            g.covariances_cholesky_ = t[9].cast<
                Eigen::Matrix<T, Eigen::Dynamic, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>();
            return g;
          }));
}

PYBIND11_MODULE(gmm_py, g)
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
