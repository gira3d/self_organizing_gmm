#pragma once

#include <stdexcept>
#include <memory>

#include <Eigen/Dense>

// to allow for debugging uninitialized matrices
#define EIGEN_INITIALIZE_MATRICES_BY_NAN

namespace sogmm
{
  class NotImplemented : public std::logic_error
  {
  public:
    NotImplemented() : std::logic_error("Function not yet implemented"){};
  };

  namespace cpu
  {
    /// @brief Container for SOGMM on the CPU using Eigen matrices.
    /// @tparam T Datatype (e.g., float, double)
    /// @tparam D Number of dimensions (e.g., 1, 2, 3, 4)
    template <typename T, uint32_t D>
    class SOGMM
    {
    public:
      using Ptr = std::shared_ptr<SOGMM<T, D>>;
      using ConstPtr = std::shared_ptr<const SOGMM<T, D>>;

      static constexpr uint32_t C = D * D;

      using Vector = Eigen::Matrix<T, -1, 1>;
      using Matrix = Eigen::Matrix<T, -1, -1, (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;

      using VectorD = Eigen::Matrix<T, D, 1>;
      using VectorC = Eigen::Matrix<T, C, 1>;
      using MatrixDD = Eigen::Matrix<T, D, D, Eigen::RowMajor>;

      using MatrixXD = Eigen::Matrix<T, -1, D, (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
      using MatrixDX = Eigen::Matrix<T, D, -1, Eigen::RowMajor>;
      using MatrixXC = Eigen::Matrix<T, -1, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;

      Vector weights_;
      MatrixXD means_;
      MatrixXC covariances_;
      MatrixXC covariances_cholesky_;
      MatrixXC precisions_cholesky_;

      uint32_t support_size_;
      uint32_t n_components_;

      /// @brief Default constructor.
      /// @details Initially there are no points in the support and no components.
      SOGMM()
      {
        support_size_ = 0;
        n_components_ = 0;
      }

      /// @brief Copy constructor.
      /// @param that SOGMM to copy from.
      SOGMM(const SOGMM &that)
      {
        this->support_size_ = that.support_size_;
        this->n_components_ = that.n_components_;

        this->weights_ = Vector::Zero(n_components_);
        this->means_ = MatrixXD::Zero(n_components_, D);
        this->covariances_ = MatrixXC::Zero(n_components_, C);
        this->covariances_cholesky_ = MatrixXC::Zero(n_components_, C);
        this->precisions_cholesky_ = MatrixXC::Zero(n_components_, C);

        this->weights_ = that.weights_;
        if (this->support_size_ > 0 && this->weights_.sum() > 0.0)
        {
          this->normalizeWeights();
        }

        this->means_ = that.means_;
        this->covariances_ = that.covariances_;
        this->precisions_cholesky_ = that.precisions_cholesky_;
        this->covariances_cholesky_ = that.covariances_cholesky_;
      }

      /// @brief Initialization with known number of components.
      /// @param n_components Number of components in the SOGMM.
      SOGMM(const uint32_t &n_components)
      {
        if (n_components <= 0)
        {
          throw std::runtime_error("Number of components should be atleast 1.");
        }

        n_components_ = n_components;

        weights_ = Vector::Zero(n_components_);
        means_ = MatrixXD::Zero(n_components_, D);
        covariances_ = MatrixXC::Zero(n_components_, C);
        covariances_cholesky_ = MatrixXC::Zero(n_components_, C);
        precisions_cholesky_ = MatrixXC::Zero(n_components_, C);
      }

      /// @brief Initialization with known SOGMM parameters.
      /// @param weights Weights of the SOGMM. Should be normalized to 1.0.
      /// @param means Means of the SOGMM.
      /// @param covariances Covariances of the SOGMM.
      /// @param support_size Number of points in the support of the SOGMM.
      SOGMM(const Vector &weights, const MatrixXD &means,
            const MatrixXC &covariances, const uint32_t &support_size)
      {
        if (support_size <= 1)
        {
          throw std::runtime_error("The support size for this SOGMM is less than or equal to 1.");
        }

        support_size_ = support_size;
        n_components_ = means.rows();

        weights_ = Vector::Zero(n_components_);
        means_ = MatrixXD::Zero(n_components_, D);
        covariances_ = MatrixXC::Zero(n_components_, C);
        precisions_cholesky_ = MatrixXC::Zero(n_components_, C);
        covariances_cholesky_ = MatrixXC::Zero(n_components_, C);

        weights_ = weights;
        means_ = means;
        covariances_ = covariances;

        updateCholesky(covariances);
      }

      void updateCholesky(const MatrixXC &covariances)
      {
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
          precisions_cholesky_.row(k) =
              Eigen::Map<VectorC>(prec_chol.data(), prec_chol.size());
        }
      }

      /// @brief Normalize the weights of this SOGMM.
      void normalizeWeights()
      {
        if (support_size_ == 0)
        {
          throw std::runtime_error("Attempted to normalize the weights when support size is zero.");
        }

        if (weights_.sum() == 0)
        {
          throw std::runtime_error("Attempted to normalize a weights array of all zeros.");
        }

        weights_.array() /= support_size_;
        weights_.array() /= weights_.sum();
      }

      /// @brief Merge the input GMM into this GMM.
      /// @param that Input GMM that needs to be merged.
      void merge(const SOGMM &that)
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
        n_components_ += that.n_components_;
      }

      SOGMM submapFromIndices(const std::vector<size_t> &indices)
      {
        unsigned int support_size = 0;
        for (size_t i = 0; i < indices.size(); i++)
        {
          support_size += support_size_ * weights_(indices[i]);
        }

        if (support_size <= 1)
        {
          return SOGMM();
        }
        else
        {
          return SOGMM(weights_(indices), means_(indices, Eigen::all), covariances_(indices, Eigen::all),
                       support_size);
        }
      }
    };

    template <typename T>
    SOGMM<T, 3> extractXpart(const SOGMM<T, 4> &input)
    {
      SOGMM<T, 3> sogmm3 = SOGMM<T, 3>(input.n_components_);

      sogmm3.weights_ = input.weights_;
      sogmm3.means_ = input.means_(Eigen::all, {0, 1, 2});

      using Matrix3 = typename SOGMM<T, 3>::MatrixDD;
      using Matrix4 = typename SOGMM<T, 4>::MatrixDD;
      using VectorC3 = typename SOGMM<T, 3>::VectorC;
      using VectorC4 = typename SOGMM<T, 4>::VectorC;
      Matrix3 temp_mat3 = Matrix3::Zero(3, 3);
      Matrix4 temp_mat4 = Matrix4::Zero(4, 4);
      VectorC4 c = VectorC4::Zero(16, 1);
      for (size_t k = 0; k < sogmm3.n_components_; k++)
      {
        // copy covariance
        c = input.covariances_.row(k);
        temp_mat4 = Eigen::Map<Matrix4>(c.data(), 4, 4);
        temp_mat3 = temp_mat4({0, 1, 2}, {0, 1, 2});
        sogmm3.covariances_.row(k) = Eigen::Map<VectorC3>(temp_mat3.data(), temp_mat3.size());
      }

      sogmm3.updateCholesky(sogmm3.covariances_);

      return sogmm3;
    }
  }
}