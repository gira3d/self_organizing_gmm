#pragma once

#include <memory>
#include <random>
#include <iostream>

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

#include <self_organizing_gmm/SOGMMCPU.h>
#include <self_organizing_gmm/EM.h>

namespace sogmm
{
  namespace cpu
  {
    template <typename T>
    class SOGMMInference
    {
    public:
      // MVN_NORM_3 = 1.0 / std::pow(std::sqrt(2.0 * M_PI), 3));
      static constexpr T MVN_NORM_3 = static_cast<T>(0.06349363593424098);

      template <uint32_t D>
      using Container = SOGMM<T, D>;
      template <uint32_t D>
      using VectorD = typename Container<D>::VectorD;
      template <uint32_t D>
      using VectorC = typename Container<D>::VectorC;
      template <uint32_t D>
      using MatrixDD = typename Container<D>::MatrixDD;
      template <uint32_t D>
      using MatrixXD = typename Container<D>::MatrixXD;
      template <uint32_t D>
      using MatrixDX = typename Container<D>::MatrixDX;
      template <uint32_t D>
      using MatrixXC = typename Container<D>::MatrixXC;

      using Vector = typename Container<4>::Vector;
      using Matrix = typename Container<4>::Matrix;

      // for Box-Muller sampling
      std::default_random_engine generator_;
      std::normal_distribution<T> normal_dist_;
      Eigen::Rand::P8_mt19937_64 urng_{42};

      SOGMMInference()
      {
        normal_dist_ =
            std::normal_distribution<T>(static_cast<T>(0.0), static_cast<T>(1.0));
      }

      template <uint32_t D>
      void sample(const Container<D> &sogmm,
                  const unsigned int &N,
                  double sigma,
                  MatrixXD<D> &samples)
      {
        unsigned int K = sogmm.n_components_;

        Eigen::Rand::MultinomialGen mgen(static_cast<int32_t>(N), sogmm.weights_);
        auto buckets = mgen.generate(urng_, 1);

        // prepare the samples matrix
        std::vector<T> x;
        x.reserve(N * D);

        while (x.size() < N * D)
        {
          T rand_val = normal_dist_(generator_);

          if (std::abs(rand_val) < sigma)
          {
            x.push_back(rand_val);
          }
        }

        samples = Eigen::Map<MatrixXD<D>>(x.data(), N, D);

        unsigned int prev_idx = 0;
        for (unsigned int k = 0; k < K; k++)
        {
          // covariance cholesky for this component
          VectorC<D> cov_chol_vector = sogmm.covariances_cholesky_.row(k);
          MatrixDD<D> L = Eigen::Map<MatrixDD<D>>(cov_chol_vector.data(), D, D);
          VectorD<D> mean = sogmm.means_.row(k);

#pragma omp parallel
          {
#pragma omp for
            for (unsigned int n = 0; n < buckets(k); n++)
            {
              VectorD<D> z = samples.row(prev_idx + n);
              VectorD<D> Lz = L * z;
              samples.row(prev_idx + n) = Lz + mean;
            }
          }
          prev_idx += buckets(k);
        }
      }

      void sampleConditional(const Container<4> &sogmm,
                             const unsigned int &N,
                             double sigma,
                             MatrixXD<3> &samples,
                             Vector &expected_values)
      {
        unsigned int K = sogmm.n_components_;

        Matrix ws = Matrix::Zero(N, K);
        Matrix ms = Matrix::Zero(N, K);

        Container<3> sogmm3 = extractXpart(sogmm);

        // parts of the mean vector
        VectorD<3> mu_kX = VectorD<3>::Zero(3);
        T mu_kli;

        // parts of the covariance matrix
        MatrixDD<4> sigma_full = MatrixDD<4>::Zero(4, 4);
        MatrixDD<3> sigma_kXX = MatrixDD<3>::Zero(3, 3);
        MatrixDD<3> sigma_kXX_inv = MatrixDD<3>::Zero(3, 3);
        VectorD<3> sigma_kXli = VectorD<3>::Zero(3);
        VectorD<3> sigma_kliX = VectorD<3>::Zero(3);
        T sigma_klili;

        VectorD<3> sigma_kliX_kXX_inv = VectorD<3>::Zero(1, 3);

        // temp variable to reduce redundant computation later
        // Linv is the inverse of (lower triangular) Cholesky decomposition of
        // sigma_kXX
        MatrixDD<3> L = MatrixDD<3>::Zero(3, 3);
        MatrixDD<3> Linv = MatrixDD<3>::Zero(3, 3);
        T cov_det, norm_factor;

        Eigen::Rand::MultinomialGen mgen(static_cast<int32_t>(N), sogmm.weights_);
        auto buckets = mgen.generate(urng_, 1);

        // prepare the samples matrix
        std::vector<T> x;
        x.reserve(N * 3);

        while (x.size() < N * 3)
        {
          T rand_val = normal_dist_(generator_);

          if (std::abs(rand_val) < sigma)
          {
            x.push_back(rand_val);
          }
        }

        samples = Eigen::Map<MatrixXD<3>>(x.data(), N, 3);

        unsigned int prev_idx = 0;
        for (unsigned int k = 0; k < K; k++)
        {
          // covariance cholesky for this component
          VectorC<3> cov_chol_vector = sogmm3.covariances_cholesky_.row(k);
          MatrixDD<3> L = Eigen::Map<MatrixDD<3>>(cov_chol_vector.data(), 3, 3);
          VectorD<3> mean = sogmm3.means_.row(k);

#pragma omp parallel
          {
#pragma omp for
            for (unsigned int n = 0; n < buckets(k); n++)
            {
              VectorD<3> z = samples.row(prev_idx + n);
              VectorD<3> Lz = L * z;
              samples.row(prev_idx + n) = Lz + mean;
            }
          }

          std::vector<int> indices;
          for (unsigned int n = 0; n < buckets(k); n++)
          {
            indices.push_back(prev_idx + n);
          }

          MatrixXD<3> X = samples(indices, Eigen::all);
          MatrixDX<3> dev = MatrixDX<3>::Zero(3, buckets(k));
          // y = L^{-1} (X - µ)^T
          MatrixDX<3> y = MatrixDX<3>::Zero(3, buckets(k));
          Vector temp = Vector::Zero(buckets(k));

          // parts of the mean vector
          mu_kX = sogmm.means_(k, {0, 1, 2});
          mu_kli = static_cast<T>(sogmm.means_(k, 3));

          // parts of the covariance matrix
          sigma_full = Eigen::Map<const MatrixDD<4>>(sogmm.covariances_.row(k).data(), 4, 4);
          sigma_kXX = sigma_full({0, 1, 2}, {0, 1, 2});
          sigma_kXli = sigma_full({0, 1, 2}, {3});
          sigma_kliX = sigma_full({3}, {0, 1, 2});
          sigma_klili = static_cast<T>(sigma_full(3, 3));

          dev = (X.rowwise() - mu_kX.transpose()).transpose();

          // begin multivariate normal computation
          cov_det = static_cast<T>(sigma_kXX.determinant());
          if (cov_det <= 0.0)
          {
            continue;
          }

          norm_factor = MVN_NORM_3 * static_cast<T>(1.0 / std::sqrt(cov_det));

          // compute the term inside the exponential
          // avoid the matrix inverse using a linear solver via Cholesky decomposition
          // solving L y = (X - µ)^T gives the solution y = L^{-1} (X - µ)^T
          // then, the term (X - µ)^T Σ^{-1} (X - µ) becomes y^T y (easy to derive)
          L = sigma_kXX.llt().matrixL();
          Linv = L.inverse();
          y = Linv * dev;
          temp = (y.array().square().colwise().sum()).array() * (-0.5);

          ws(indices, k) = ((temp.array().exp()) * norm_factor);
          ws(indices, k) = sogmm.weights_(k) * ws(indices, k);

          sigma_kXX_inv = (Linv.transpose()) * Linv;
          sigma_kliX_kXX_inv = sigma_kliX.transpose() * sigma_kXX_inv;

          ms(indices, k) = ((sigma_kliX_kXX_inv.transpose() * dev).array() + mu_kli).transpose();

          prev_idx += buckets(k);
        }

        // clamp ws to zero
        ws = (ws.array() < 1e-6).select(0.0, ws);

        Vector ws_sums = Vector::Zero(N);
        ws_sums << ws.rowwise().sum();

        // normalize in place
        ws.array().colwise() /= ws_sums.array();

        // nan to zeros
        ws = (ws.array().isFinite()).select(ws, static_cast<T>(0.0));

        // expected values
        expected_values = Vector::Zero(N);
        expected_values = (ws.array() * ms.array()).rowwise().sum();
      }

      void colorConditional(const MatrixXD<3> &X,
                            const Container<4> &sogmm,
                            Vector &expected_values,
                            Vector &uncerts)
      {
        unsigned int N = X.rows();
        unsigned int K = sogmm.n_components_;

        Matrix ws = Matrix::Zero(N, K);
        Matrix ms = Matrix::Zero(N, K);
        Vector vars = Vector::Zero(K);

        MatrixDX<3> dev = MatrixDX<3>::Zero(3, N);

        // parts of the mean vector
        VectorD<3> mu_kX = VectorD<3>::Zero(3);
        T mu_kli;

        // parts of the covariance matrix
        MatrixDD<4> sigma = MatrixDD<4>::Zero(4, 4);
        MatrixDD<3> sigma_kXX = MatrixDD<3>::Zero(3, 3);
        MatrixDD<3> sigma_kXX_inv = MatrixDD<3>::Zero(3, 3);
        VectorD<3> sigma_kXli = VectorD<3>::Zero(3);
        VectorD<3> sigma_kliX = VectorD<3>::Zero(3);
        T sigma_klili;

        VectorD<3> sigma_kliX_kXX_inv = VectorD<3>::Zero(1, 3);

        // temp variable to reduce redundant computation later
        // Linv is the inverse of (lower triangular) Cholesky decomposition of
        // sigma_kXX
        MatrixDD<3> L = MatrixDD<3>::Zero(3, 3);
        MatrixDD<3> Linv = MatrixDD<3>::Zero(3, 3);
        // y = L^{-1} (X - µ)^T
        MatrixDX<3> y = MatrixDX<3>::Zero(3, N);

        Vector temp = Vector::Zero(N);
        T cov_det, norm_factor;

        for (int k = 0; k < K; k++)
        {
          // parts of the mean vector
          mu_kX = sogmm.means_(k, {0, 1, 2});
          mu_kli = static_cast<T>(sogmm.means_(k, 3));

          // parts of the covariance matrix
          sigma = Eigen::Map<const MatrixDD<4>>(sogmm.covariances_.row(k).data(), 4, 4);
          sigma_kXX = sigma({0, 1, 2}, {0, 1, 2});
          sigma_kXli = sigma({0, 1, 2}, {3});
          sigma_kliX = sigma({3}, {0, 1, 2});
          sigma_klili = static_cast<T>(sigma(3, 3));

          dev = (X.rowwise() - mu_kX.transpose()).transpose();

          // begin multivariate normal computation
          cov_det = static_cast<T>(sigma_kXX.determinant());
          if (cov_det <= 0.0)
          {
            continue;
          }

          norm_factor = MVN_NORM_3 * static_cast<T>(1.0 / std::sqrt(cov_det));

          // compute the term inside the exponential
          // avoid the matrix inverse using a linear solver via Cholesky decomposition
          // solving L y = (X - µ)^T gives the solution y = L^{-1} (X - µ)^T
          // then, the term (X - µ)^T Σ^{-1} (X - µ) becomes y^T y (easy to derive)
          L = sigma_kXX.llt().matrixL();
          Linv = L.inverse();
          y = Linv * dev;
          temp = (y.array().square().colwise().sum()).array() * (-0.5);

          ws.col(k) = ((temp.array().exp()) * norm_factor);
          if (!ws.col(k).array().isFinite().any())
          {
            std::cerr << "k " << k << " cov_det " << cov_det << std::endl;
            std::cerr << "k " << k << " Linv " << Linv << std::endl;
          }
          ws.col(k) = sogmm.weights_(k) * ws.col(k);

          sigma_kXX_inv = (Linv.transpose()) * Linv;
          sigma_kliX_kXX_inv = sigma_kliX.transpose() * sigma_kXX_inv;

          ms.col(k) = ((sigma_kliX_kXX_inv.transpose() * dev).array() + mu_kli).transpose();

          // TODO: uncomment if uncertainties are needed
          // vars(k) = (sigma_klili - (sigma_kliX_kXX_inv.transpose() * sigma_kXli).value());
        }

        // clamp ws to zero
        ws = (ws.array() < 1e-6).select(0.0, ws);

        Vector ws_sums = Vector::Zero(N);
        ws_sums << ws.rowwise().sum();

        // normalize in place
        ws.array().colwise() /= ws_sums.array();

        // nan to zeros
        ws = (ws.array().isFinite()).select(ws, static_cast<T>(0.0));

        // expected values
        expected_values = Vector::Zero(N);
        expected_values = (ws.array() * ms.array()).rowwise().sum();

        // uncertainty
        uncerts = Vector::Zero(N);
        // TODO: uncomment if uncertainties are needed
        // uncerts = (ws.array() * (ms.array().square().rowwise() + vars.transpose().array())).rowwise().sum();
        // uncerts = uncerts.array() - expected_values.array().square();
      }

      MatrixXD<4> generatePointCloud4D(const Container<4> &sogmm,
                                       const unsigned int &N,
                                       double sigma)
      {
        MatrixXD<4> samples;
        sample<4>(sogmm, N, sigma, samples);
        return samples;
      }

      MatrixXD<3> generatePointCloud3D(const Container<4> &sogmm,
                                       const unsigned int &N,
                                       double sigma)
      {
        // Extract the (x, y, z) part from SOGMM
        Container<3> sogmm3 = extractXpart(sogmm);

        MatrixXD<3> samples;
        sample<3>(sogmm3, N, sigma, samples);
        return samples;
      }

      Vector score4D(const MatrixXD<4> &X, const Container<4> &sogmm)
      {
        return EM<T, 4>::scoreSamples(X, sogmm);
      }

      Vector score3D(const MatrixXD<3> &X, const Container<4> &sogmm)
      {
        // Extract the (x, y, z) part from SOGMM
        Container<3> sogmm3 = extractXpart(sogmm);

        return EM<T, 3>::scoreSamples(X, sogmm3);
      }

      MatrixXD<4> reconstruct(const Container<4> sogmm,
                              const unsigned int &N,
                              double sigma)
      {
        MatrixXD<3> samples = generatePointCloud3D(sogmm, N, sigma);

        Vector E, V;
        colorConditional(samples, sogmm, E, V);

        MatrixXD<4> ret = MatrixXD<4>::Zero(N, 4);
        ret(Eigen::all, {0, 1, 2}) = samples;
        ret(Eigen::all, {3}) = E;

        return ret;
      }

      MatrixXD<4> colorQuery(const Container<4> sogmm,
                             const MatrixXD<3> &samples,
                             const unsigned int &N)
      {
        Vector E, V;
        colorConditional(samples, sogmm, E, V);

        MatrixXD<4> ret = MatrixXD<4>::Zero(N, 4);
        ret(Eigen::all, {0, 1, 2}) = samples;
        ret(Eigen::all, {3}) = E;

        return ret;
      }

      MatrixXD<4> reconstructFast(const Container<4> sogmm,
                                  const unsigned int &N,
                                  double sigma)
      {
        MatrixXD<3> samples;
        Vector E;
        sampleConditional(sogmm, N, sigma, samples, E);

        MatrixXD<4> ret = MatrixXD<4>::Zero(N, 4);
        ret(Eigen::all, {0, 1, 2}) = samples;
        ret(Eigen::all, {3}) = E;

        return ret;
      }
    };
  }
}