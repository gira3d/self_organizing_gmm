#pragma once

#include <memory>
#include <random>

#include <Eigen/Dense>

#include <self_organizing_gmm/SOGMMCPU.h>

// to allow for debugging uninitialized matrices
#define EIGEN_INITIALIZE_MATRICES_BY_NAN

#define LOG_2_M_PI 1.83788

namespace sogmm
{
  namespace cpu
  {
    /// @brief Class to run Expectation-Maximization on CPU.
    /// @tparam T Datatype (e.g., float, double)
    /// @tparam D Dimension of the data (e.g., 1, 2, 3, 4)
    template <typename T, uint32_t D>
    class EM
    {
    public:
      static constexpr uint32_t C = D * D;

      using Ptr = std::shared_ptr<EM<T, D>>;
      using ConstPtr = std::shared_ptr<const EM<T, D>>;

      using Container = SOGMM<T, D>;

      using Vector = typename Container::Vector;
      using Matrix = typename Container::Matrix;
      using VectorD = typename Container::VectorD;
      using VectorC = typename Container::VectorC;
      using MatrixDD = typename Container::MatrixDD;
      using MatrixXD = typename Container::MatrixXD;
      using MatrixDX = typename Container::MatrixDX;
      using MatrixXC = typename Container::MatrixXC;

      EM() : EM(1e-3, 1e-6, 100)
      {
      }

      EM(T tol, T reg_covar, unsigned int max_iter)
      {
        tol_ = tol;
        reg_covar_ = reg_covar;
        max_iter_ = max_iter;
      }

      ~EM()
      {
      }

      static void computeLogDetCholesky(const MatrixXC &matrix_chol, Vector &log_det_chol)
      {
        MatrixDD chol = MatrixDD::Zero(D, D);
        VectorC c = VectorC::Zero(C, 1);
        for (unsigned k = 0; k < matrix_chol.rows(); k++)
        {
          c = matrix_chol.row(k);
          chol = Eigen::Map<MatrixDD>(c.data(), D, D);
          log_det_chol(k) = chol.diagonal().array().log().sum();
        }
      }

      static Matrix estimateLogGaussianProb(const MatrixXD &X,
                                            const Container &sogmm)
      {
        // Eq. 3.14 from
        // http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-108.pdf#page=39

        unsigned int K = sogmm.n_components_;

        Vector log_det = Vector::Zero(K);
        computeLogDetCholesky(sogmm.precisions_cholesky_, log_det);

        Matrix log_prob = Matrix::Zero(X.rows(), K);

        MatrixDD prec_chol = MatrixDD::Zero(D, D);
        Matrix y = Matrix::Zero(X.rows(), D);
        VectorC p = VectorC::Zero(C, 1);
        for (unsigned k = 0; k < K; k++)
        {
          p = sogmm.precisions_cholesky_.row(k);
          prec_chol = Eigen::Map<MatrixDD>(p.data(), D, D);
          y = (X * prec_chol).rowwise() - (sogmm.means_.row(k) * prec_chol);
          log_prob.col(k) = y.array().square().rowwise().sum();
        }

        return (-0.5 * ((D * LOG_2_M_PI) + log_prob.array())).rowwise() +
               log_det.transpose().array();
      }

      static void estimateWeightedLogProb(const MatrixXD &X,
                                          const Container &sogmm,
                                          Matrix &weighted_log_prob)
      {
        weighted_log_prob = estimateLogGaussianProb(X, sogmm).array().rowwise() +
                            sogmm.weights_.transpose().array().log();
      }

      static Vector logSumExpCols(const Matrix &A)
      {
        Vector amax = A.rowwise().maxCoeff();
        Vector logsumexp = (A.colwise() - amax).array().exp().rowwise().sum().log();

        return logsumexp + amax;
      }

      inline void eStep(const MatrixXD &X, const Container &sogmm,
                        T &lower_bound, Matrix &log_resp)
      {
        estimateWeightedLogProb(X, sogmm, log_resp);
        Vector log_prob_norm = logSumExpCols(log_resp);

        log_resp.colwise() -= log_prob_norm;
        lower_bound = log_prob_norm.array().mean();
      }

      inline void mStep(const MatrixXD &X, const Matrix &resp, Container &sogmm)
      {
        unsigned int K = sogmm.n_components_;

        sogmm.weights_ = Vector::Zero(K);
        sogmm.weights_ = resp.colwise().sum() +
                         Matrix::Constant(1, K, 10 * std::numeric_limits<T>::epsilon());

        sogmm.means_ = (resp.transpose() * X).array().colwise() / sogmm.weights_.array();
        sogmm.covariances_ = MatrixXC::Zero(K, C);
#pragma omp parallel
        {
#pragma omp for
          for (unsigned int k = 0; k < K; k++)
          {
            MatrixXD diff = MatrixXD::Zero(X.rows(), D);
            MatrixDX diff_temp = MatrixDX::Zero(D, X.rows());
            MatrixDD cov = MatrixDD::Zero(D, D);
            diff = X.rowwise() - sogmm.means_.row(k);
            diff_temp = (diff.transpose().array().rowwise() *
                         resp.col(k).transpose().array());
            cov = diff_temp * diff / sogmm.weights_(k);
            cov.diagonal().array() += reg_covar_;
            sogmm.covariances_.row(k) = Eigen::Map<VectorC>(cov.data(), cov.size());
          }
        }

        sogmm.weights_.array() /= X.rows();
        sogmm.weights_.array() /= sogmm.weights_.sum();

        sogmm.updateCholesky(sogmm.covariances_);
      }

      inline bool fit(const MatrixXD &X, const Matrix &resp, Container &sogmm)
      {
        unsigned int K = sogmm.n_components_;

        if (X.rows() <= 1)
        {
          throw std::runtime_error("fit: number of samples should be greater than 1.");
        }

        if (K <= 0)
        {
          throw std::runtime_error("fit: number of components should be greater than 0.");
        }

        if (X.rows() < K)
        {
          throw std::runtime_error("fit: number of components is " +
                                   std::to_string(K) +
                                   ". It should be strictly smaller than the "
                                   "number of points: " +
                                   std::to_string(X.rows()));
        }

        mStep(X, resp, sogmm);

        T lower_bound = -std::numeric_limits<T>::infinity();
        Matrix log_resp = Matrix::Zero(X.rows(), K);
        for (unsigned int n_iter = 0; n_iter <= max_iter_; n_iter++)
        {
          T prev_lower_bound = lower_bound;

          // E step
          eStep(X, sogmm, lower_bound, log_resp);

          // M step
          mStep(X, log_resp.array().exp(), sogmm);

          // convergence check
          T change = lower_bound - prev_lower_bound;
          if (!std::isinf(change) && std::abs(change) < tol_)
          {
            converged_ = true;
            break;
          }
        }

        if (converged_)
        {
          return true;
        }
        else
        {
          return false;
        }
      }

      static Vector scoreSamples(const MatrixXD &X, const Container &sogmm)
      {
        Matrix scores = Matrix::Zero(X.rows(), sogmm.n_components_);
        estimateWeightedLogProb(X, sogmm, scores);
        return logSumExpCols(scores);
      }

      static T score(const MatrixXD &X, const Container &sogmm)
      {
        return static_cast<T>(scoreSamples(X, sogmm).array().mean());
      }

      bool converged_ = false;

      T tol_;
      T reg_covar_;
      unsigned int max_iter_;
    };
  }
}