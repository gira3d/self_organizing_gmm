#pragma once

#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

#define DBG_MACRO_NO_WARNING
#define DBG_MACRO_DISABLE
#include "utils/dbg.h"

#include "TimeProfiler.h"

// to allow for debugging uninitialized matrices
#define EIGEN_INITIALIZE_MATRICES_BY_NAN

#define LOG_2_M_PI 1.83788

// T -- datatypes (usually float or double)
// D -- dimension of the data
template <typename T, uint32_t D>
class GMM
{
public:
  static constexpr uint32_t C = D * D;
  // MVN_NORM_3 = 1.0 / std::pow(std::sqrt(2.0 * M_PI), 3));
  static constexpr T MVN_NORM_3 = static_cast<T>(0.06349363593424098);

  using Ptr = std::shared_ptr<GMM<T, D>>;
  using ConstPtr = std::shared_ptr<const GMM<T, D>>;

  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using VectorD = Eigen::Matrix<T, D, 1>;
  using VectorC = Eigen::Matrix<T, C, 1>;

  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                               (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
  using MatrixXD = Eigen::Matrix<T, Eigen::Dynamic, D,
                                 (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
  using MatrixDX = Eigen::Matrix<T, D, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXC = Eigen::Matrix<T, Eigen::Dynamic, C,
                                 (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
  using MatrixDD = Eigen::Matrix<T, D, D, Eigen::RowMajor>;

  GMM() : GMM(1, 1e-3, 1e-6, 100)
  {
  }

  GMM(unsigned int n_components) : GMM(n_components, 1e-3, 1e-6, 100)
  {
  }

  GMM(unsigned int n_components, T tol, T reg_covar, unsigned int max_iter)
  {
    n_components_ = n_components;
    tol_ = tol;
    reg_covar_ = reg_covar;
    max_iter_ = max_iter;

    initialize(false, "gmm_cpu", "stats.csv");
  }

  GMM(unsigned int n_components, const bool save_stats,
      const std::string& stats_dir, const std::string& stats_file)
    : GMM(n_components, 1e-3, 1e-6, 100, save_stats, stats_dir, stats_file)
  {
  }

  GMM(unsigned int n_components, T tol, T reg_covar, unsigned int max_iter,
      const bool save_stats, const std::string& stats_dir,
      const std::string& stats_file)
  {
    n_components_ = n_components;
    tol_ = tol;
    reg_covar_ = reg_covar;
    max_iter_ = max_iter;

    initialize(save_stats, stats_dir, stats_file);
  }

  void initialize(const bool save_stats, const std::string& stats_dir,
                  const std::string& stats_file)
  {
    tp_ = TimeProfiler();
    if (save_stats)
    {
      tp_.save(stats_dir, stats_file);
    }

    weights_ = Vector::Zero(n_components_);
    means_ = MatrixXD::Zero(n_components_, D);
    covariances_ = MatrixXC::Zero(n_components_, C);
    covariances_cholesky_ = MatrixXC::Zero(n_components_, C);

    normal_dist_ =
        std::normal_distribution<T>(static_cast<T>(0.0), static_cast<T>(1.0));
  }

  ~GMM()
  {
  }

  inline Vector computeLogDetCholesky(const MatrixXC& matrix_chol)
  {
    Vector log_det_chol = Vector::Zero(n_components_, 1);

    MatrixDD chol = MatrixDD::Zero(D, D);
    VectorC c = VectorC::Zero(C, 1);
    for (unsigned k = 0; k < n_components_; k++)
    {
      c = matrix_chol.row(k);
      chol = Eigen::Map<MatrixDD>(c.data(), D, D);
      log_det_chol(k) = chol.diagonal().array().log().sum();
    }

    return log_det_chol;
  }

  inline MatrixXC computeCholesky(const MatrixXC& covariances)
  {
    MatrixXC prec_chols = Matrix::Zero(n_components_, C);

    MatrixDD cov = MatrixDD::Zero(D, D);
    MatrixDD cov_chol = MatrixDD::Zero(D, D);
    MatrixDD prec_chol = MatrixDD::Zero(D, D);
    VectorC c = VectorC::Zero(C, 1);
    for (unsigned k = 0; k < n_components_; k++)
    {
      c = covariances.row(k);
      cov = Eigen::Map<MatrixDD>(c.data(), D, D);
      cov_chol = cov.llt().matrixL();
      covariances_cholesky_.row(k) =
          Eigen::Map<VectorC>(cov_chol.data(), cov_chol.size());
      prec_chol = (cov_chol.transpose() * cov_chol)
                      .ldlt()
                      .solve(cov_chol.transpose() * MatrixDD::Identity(D, D))
                      .transpose();
      prec_chols.row(k) =
          Eigen::Map<VectorC>(prec_chol.data(), prec_chol.size());
    }

    return prec_chols;
  }

  inline Matrix estimateLogGaussianProb(const MatrixXD& X,
                                        const MatrixXD& means,
                                        const MatrixXC& precisions_chol)
  {
    // Eq. 3.14 from
    // http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-108.pdf#page=39

    unsigned int n_samples = X.rows();

    Vector log_det = Vector::Zero(n_components_);
    log_det = computeLogDetCholesky(precisions_chol);

    Matrix log_prob = Matrix::Zero(n_samples, n_components_);

    MatrixDD prec_chol = MatrixDD::Zero(D, D);
    Matrix y = Matrix::Zero(n_samples, D);
    VectorC p = VectorC::Zero(C, 1);
    for (unsigned k = 0; k < n_components_; k++)
    {
      p = precisions_chol.row(k);
      prec_chol = Eigen::Map<MatrixDD>(p.data(), D, D);
      y = (X * prec_chol).rowwise() - (means.row(k) * prec_chol);
      log_prob.col(k) = y.array().square().rowwise().sum();
    }

    return (-0.5 * ((D * LOG_2_M_PI) + log_prob.array())).rowwise() +
           log_det.transpose().array();
  }

  inline Matrix estimateLogProb(const MatrixXD& X)
  {
    return estimateLogGaussianProb(X, means_, precisions_cholesky_);
  }

  inline Matrix estimateWeightedLogProb(const MatrixXD& X)
  {
    return estimateLogProb(X).array().rowwise() +
           weights_.transpose().array().log();
  }

  inline Vector logSumExpCols(const Matrix& A)
  {
    Vector amax = A.rowwise().maxCoeff();
    Vector logsumexp = (A.colwise() - amax).array().exp().rowwise().sum().log();

    return logsumexp + amax;
  }

  inline std::pair<Matrix, Matrix> estimateLogProbResp(const MatrixXD& X)
  {
    Matrix weighted_log_prob = estimateWeightedLogProb(X);
    Vector log_prob_norm = logSumExpCols(weighted_log_prob);
    Matrix log_resp = weighted_log_prob.colwise() - log_prob_norm;

    return std::make_pair(log_prob_norm, log_resp);
  }

  inline std::pair<T, Matrix> eStep(const MatrixXD& X)
  {
    tp_.tic("eStep");
    auto log_prob_resp = estimateLogProbResp(X);
    tp_.toc("eStep");
    return std::make_pair(log_prob_resp.first.array().mean(),
                          log_prob_resp.second);
  }

  inline std::tuple<Vector, Matrix, MatrixXC>
  estimateGaussianParameters(const MatrixXD& X, const Matrix& resp)
  {
    Vector nk = Vector::Zero(n_components_);
    nk = resp.colwise().sum() +
         Matrix::Constant(1, n_components_,
                          10 * std::numeric_limits<T>::epsilon());

    Matrix means = (resp.transpose() * X).array().colwise() / nk.array();

    unsigned int n_samples = X.rows();
    MatrixXC covariances = MatrixXC::Zero(n_components_, C);
#pragma omp parallel
    {
#pragma omp for
      for (unsigned int k = 0; k < n_components_; k++)
      {
        MatrixXD diff = MatrixXD::Zero(n_samples, D);
        MatrixDX diff_temp = MatrixDX::Zero(D, n_samples);
        MatrixDD cov = MatrixDD::Zero(D, D);
        diff = X.rowwise() - means.row(k);
        diff_temp = (diff.transpose().array().rowwise() *
                     resp.col(k).transpose().array());
        cov = diff_temp * diff / nk(k);
        cov.diagonal().array() += reg_covar_;
        covariances.row(k) = Eigen::Map<VectorC>(cov.data(), cov.size());
      }
    }

    return std::make_tuple(nk, means, covariances);
  }

  inline void mStep(const MatrixXD& X, const Matrix& log_resp)
  {
    tp_.tic("mStep");
    unsigned int n_samples = X.rows();

    resp_ = log_resp.array().exp();

    auto gmm_model_params =
        estimateGaussianParameters(X, log_resp.array().exp());

    weights_ = std::get<0>(gmm_model_params);
    weights_.array() /= n_samples;
    weights_.array() /= weights_.sum();

    means_ = std::get<1>(gmm_model_params);
    covariances_ = std::get<2>(gmm_model_params);
    precisions_cholesky_ = computeCholesky(covariances_);
    tp_.toc("mStep");
  }

  inline bool fit(const MatrixXD& X, const Matrix& resp)
  {
    tp_.tic("fit");
    unsigned int n_samples = X.rows();

    if (n_samples <= 1)
    {
      throw std::runtime_error("fit: number of samples is " +
                               std::to_string(n_samples) +
                               ", it should be greater than 1.");
    }

    if (n_components_ <= 1)
    {
      throw std::runtime_error("fit: number of components is " +
                               std::to_string(n_components_) +
                               ", it should be greater than 1.");
    }

    auto gmm_model_params = estimateGaussianParameters(X, resp);
    weights_ = std::get<0>(gmm_model_params);
    means_ = std::get<1>(gmm_model_params);
    covariances_ = std::get<2>(gmm_model_params);
    precisions_cholesky_ = computeCholesky(covariances_);

    T lower_bound = -std::numeric_limits<T>::infinity();
    for (unsigned int n_iter = 0; n_iter <= max_iter_; n_iter++)
    {
      T prev_lower_bound = lower_bound;

      // E step
      auto estep_output = eStep(X);
      T log_prob_norm = estep_output.first;
      Matrix log_resp = estep_output.second;

      // M step
      mStep(X, log_resp);
      lower_bound = log_prob_norm;

      // convergence check
      T change = lower_bound - prev_lower_bound;
      dbg(n_iter, change);
      if (!std::isinf(change) && std::abs(change) < tol_)
      {
        converged_ = true;
        break;
      }
    }

    tp_.toc("fit");
    if (converged_)
    {
      support_size_ = n_samples;
      return true;
    }
    else
    {
      return false;
    }
  }

  // Box-Muller method sampling of Gaussian distributions
  inline MatrixXD sample(const unsigned int& n_samples, double sigma = 3.0)
  {
    Eigen::Rand::MultinomialGen mgen(static_cast<int32_t>(n_samples), weights_);
    auto buckets = mgen.generate(urng_, 1);

    // prepare the samples matrix
    std::vector<T> x;
    x.reserve(n_samples * D);

    while (x.size() < n_samples * D)
    {
      T rand_val = normal_dist_(generator_);

      if (std::abs(rand_val) < sigma)
      {
        x.push_back(rand_val);
      }
    }

    MatrixXD samples = Eigen::Map<MatrixXD>(x.data(), n_samples, D);

    unsigned int prev_idx = 0;
    for (unsigned int k = 0; k < n_components_; k++)
    {
      // covariance cholesky for this component
      VectorC cov_chol_vector = covariances_cholesky_.row(k);
      MatrixDD L = Eigen::Map<MatrixDD>(cov_chol_vector.data(), D, D);
      VectorD mean = means_.row(k);

#pragma omp parallel
      {
#pragma omp for
        for (unsigned int n = 0; n < buckets(k); n++)
        {
          VectorD z = samples.row(prev_idx + n);
          VectorD Lz = L * z;
          samples.row(prev_idx + n) = Lz + mean;
        }
      }
      prev_idx += buckets(k);
    }

    return samples;
  }

  inline Vector scoreSamples(const MatrixXD& X)
  {
    return logSumExpCols(estimateWeightedLogProb(X));
  }

  inline T score(const MatrixXD& X)
  {
    return static_cast<T>(scoreSamples(X).array().mean());
  }

  void mvnPdf(const Matrix& X, const Matrix& mu, const Matrix& sigma,
              const Matrix& Xminusmean, int k, Matrix& probs, Matrix& Linv,
              Matrix& y)
  {
    // compute determinant of covariance matrix.
    T cov_det = static_cast<T>(sigma.determinant());

    // handle the case when the determinant is too low.
    // if (cov_det < reg_covar_ * reg_covar_)
    // {
    //   cov_det = reg_covar_ * reg_covar_;
    // }

    // compute the norm factor
    T norm_factor = MVN_NORM_3 * static_cast<T>(1.0 / std::sqrt(cov_det));

    // compute the term inside the exponential
    // avoid the matrix inverse using a linear solver via Cholesky decomposition
    // solving L y = (X - µ)^T gives the solution y = L^{-1} (X - µ)^T
    // then, the term (X - µ)^T Σ^{-1} (X - µ) becomes y^T y (easy to derive)
    Matrix L = sigma.llt().matrixL();
    Linv = L.inverse();
    y = Linv * Xminusmean;
    Matrix temp = (y.array().square().colwise().sum()).array() * (-0.5);

    probs.col(k) = ((temp.array().exp()) * norm_factor).transpose();
  }

  std::tuple<Matrix, Matrix, Matrix> colorConditional(const Matrix& X)
  {
    int N = X.rows();

    Matrix ws = Matrix::Zero(N, n_components_);
    Matrix ms = Matrix::Zero(N, n_components_);
    Vector vars = Vector::Zero(n_components_);

    Matrix dev = Matrix::Zero(N, D - 1);

    // parts of the mean vector
    Matrix mu_kX = Matrix::Zero(D - 1, 1);
    T mu_kli;

    // parts of the covariance matrix
    MatrixDD sigma = Matrix::Zero(D, D);
    Matrix sigma_kXX = Matrix::Zero(D - 1, D - 1);
    Matrix sigma_kXX_inv = Matrix::Zero(D - 1, D - 1);
    Matrix sigma_kXli = Matrix::Zero(D - 1, 1);
    Matrix sigma_kliX = Matrix::Zero(1, D - 1);
    T sigma_klili;

    Matrix sigma_kliX_kXX_inv = Matrix::Zero(1, D - 1);

    // temp variable to reduce redundant computation later
    // Linv is the inverse of (lower triangular) Cholesky decomposition of
    // sigma_kXX
    Matrix Linv = Matrix::Zero(D - 1, D - 1);
    // y = L^{-1} (X - µ)^T
    Matrix y = Matrix::Zero(N, n_components_);

    for (int k = 0; k < n_components_; k++)
    {
      // parts of the mean vector
      mu_kX = means_(k, { 0, 1, 2 });
      mu_kli = static_cast<T>(means_(k, 3));

      // parts of the covariance matrix
      sigma = Eigen::Map<MatrixDD>(covariances_.row(k).data(), D, D);
      sigma_kXX = sigma({ 0, 1, 2 }, { 0, 1, 2 });
      sigma_kXli = sigma({ 0, 1, 2 }, { 3 });
      sigma_kliX = sigma({ 3 }, { 0, 1, 2 });
      sigma_klili = static_cast<T>(sigma(3, 3));

      dev = (X.rowwise() - mu_kX(0, Eigen::all)).transpose();

      mvnPdf(X, mu_kX, sigma_kXX, dev, k, ws, Linv, y);
      ws.col(k) = weights_(k) * ws.col(k);

      sigma_kXX_inv = (Linv.transpose()) * Linv;
      sigma_kliX_kXX_inv = sigma_kliX * sigma_kXX_inv;

      ms.col(k) = ((sigma_kliX_kXX_inv * dev).array() + mu_kli).transpose();
      vars(k) = (sigma_klili - (sigma_kliX_kXX_inv * sigma_kXli).value());
    }

    // clamp ws to zero
    ws = (ws.array() < reg_covar_).select(0.0, ws);

    Vector ws_sums = Vector::Zero(N);
    ws_sums << ws.rowwise().sum();

    // normalize in place
    ws.array().colwise() /= ws_sums.array();

    // nan to zeros
    ws = (ws.array().isFinite()).select(ws, static_cast<T>(0.0));

    // expected values
    Vector expected_values = Vector::Zero(N);
    expected_values = (ws.array() * ms.array()).rowwise().sum();

    // uncertainty
    Vector uncerts = Vector::Zero(N);
    uncerts = (ws.array() * (ms.array().square().rowwise() + vars.transpose().array())).rowwise().sum();
    uncerts = uncerts.array() - expected_values.array().square();

    return std::make_tuple(ws, expected_values, uncerts);
  }

  inline void merge(const GMM& that)
  {
    Vector new_weights = Vector::Zero(weights_.rows() + that.weights_.rows());
    MatrixXD new_means =
        MatrixXD::Zero(means_.rows() + that.means_.rows(), means_.cols());
    MatrixXC new_covariances = MatrixXC::Zero(
        covariances_.rows() + that.covariances_.rows(), covariances_.cols());
    MatrixXC new_precisions_cholesky = MatrixXC::Zero(
        precisions_cholesky_.rows() + that.precisions_cholesky_.rows(),
        precisions_cholesky_.cols());
    MatrixXC new_covariances_cholesky = MatrixXC::Zero(
        covariances_cholesky_.rows() + that.covariances_cholesky_.rows(),
        covariances_cholesky_.cols());

    new_weights << weights_.array() * support_size_,
        that.weights_.array() * that.support_size_;
    new_weights.array() /= (support_size_ + that.support_size_);
    new_weights.array() /= new_weights.sum();

    new_means << means_, that.means_;
    new_covariances << covariances_, that.covariances_;
    new_precisions_cholesky << precisions_cholesky_, that.precisions_cholesky_;
    new_covariances_cholesky << covariances_cholesky_,
        that.covariances_cholesky_;

    weights_ = new_weights;
    means_ = new_means;
    covariances_ = new_covariances;
    precisions_cholesky_ = new_precisions_cholesky;
    covariances_cholesky_ = new_covariances_cholesky;
    support_size_ += that.support_size_;
    converged_ = true;
    n_components_ += that.n_components_;
  }

  Vector weights_;
  MatrixXD means_;
  MatrixXC covariances_;
  MatrixXC precisions_cholesky_;
  unsigned int support_size_;
  bool converged_ = false;

  unsigned int n_components_;
  T tol_;
  T reg_covar_;
  unsigned int max_iter_;

  // for Box-Muller sampling
  MatrixXC covariances_cholesky_;
  std::default_random_engine generator_;
  std::normal_distribution<T> normal_dist_;
  Eigen::Rand::P8_mt19937_64 urng_{42};

  // for python bindings/debugging
  Matrix resp_;
  TimeProfiler tp_;
};
