#pragma once

#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>

// to allow for debugging uninitialized matrices
#define EIGEN_INITIALIZE_MATRICES_BY_NAN

namespace sogmm
{
  /// @brief Class to compute initial responsibility matrix for EM.
  /// @details For initial responsibilities, we use the steps 1a -- 1c of the
  /// algorithm from: Arthur, D. and Vassilvitskii, S.  "k-means++: the
  /// advantages ofcareful seeding". ACM-SIAM symposium on Discrete algorithms.
  /// 2007
  /// @author Kshitij Goel
  /// @tparam T Datatype (e.g., float, double)
  /// @tparam D Dimensions of the data (e.g., 1, 2, 3, 4)
  template <typename T, uint32_t D>
  class KInit
  {
  public:
    using Ptr = std::shared_ptr<KInit<T, D>>;
    using ConstPtr = std::shared_ptr<const KInit<T, D>>;

    using Vector = Eigen::Matrix<T, -1, 1>;
    using Matrix = Eigen::Matrix<T, -1, -1, (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;

    KInit()
    {
    }

    ~KInit()
    {
    }

    /// @brief Compute cumulative sum of the input matrix.
    /// @param X input matrix.
    /// @return matrix Y, same in size as X
    inline Matrix cumSum(const Matrix &X)
    {
      Matrix Y = X;

      for (unsigned int i = 1; i < X.cols(); i++)
      {
        Y(Eigen::all, i) += Y(Eigen::all, i - 1);
      }

      return Y;
    }

    /// @brief Recursively find index of the ceiling for the needle in haystack
    /// @param haystack A matrix containing many points
    /// @param low Lower index
    /// @param high Higher index
    /// @param needle Query
    /// @return Index that is ceiling of the query (needle)
    inline int ceilSearch(const Matrix &haystack, int low, int high, T needle)
    {
      // if needle is smaller than or equal to the first element in haystack,
      // then return the first element
      if (needle <= haystack(low))
        return low;

      // if needle is greater than the last element,
      // then return the last element
      if (needle > haystack(high))
        return high;

      // get the index of the middle element
      int mid = int((low + high) / 2);

      // if needle is the same as the middle element, return mid
      if (needle == haystack(mid))
      {
        return mid;
      }
      else if (needle > haystack(mid))
      {
        // if needle is greater than haystack(mid),
        // then either haystack(mid+1) is ceiling of needle
        // of ceiling lies in haystack(mid+1...high)
        if (mid + 1 <= high && needle <= haystack(mid + 1))
          return mid + 1;
        else
          return ceilSearch(haystack, mid + 1, high, needle);
      }
      else
      {
        // if needle is smaller than haystack(mid),
        // then either haystack(mid) is ceiling of needle
        // or ceiling lies in haystack(low...mid-1)
        if (mid - 1 >= low && needle > haystack(mid - 1))
          return mid;
        else
          return ceilSearch(haystack, low, mid - 1, needle);
      }
    }

    /// @brief Find indices where elements should be inserted to maintain order.
    /// @param a Input array.
    /// @param v Values to insert into a.
    /// @return Array of insertion points with the same shape as v.
    inline std::vector<unsigned int> searchSorted(const Matrix &a,
                                                  const Matrix &v)
    {
      const unsigned int a_size = a.cols();
      const unsigned int v_size = v.rows();

      std::vector<unsigned int> ret;

      for (unsigned int i = 0; i < v_size; i++)
      {
        ret.push_back(ceilSearch(a, 0, a_size - 1, v(i)));
      }

      return ret;
    }

    /// @brief Square of Euclidean distances between points in two input matrices.
    /// @param X Input matrix 1 of shape (N, D).
    /// @param Y Input matrix 2 of shape (K, D).
    /// @return A matrix of shape (N, K) with squared distances.
    Matrix euclideanDistancesSq(const Matrix &X, const Matrix &Y)
    {
      const int N = X.rows();
      const int K = Y.rows();

      Matrix Dist;

      Matrix XX, YY, XY;
      XX = Matrix::Zero(N, 1);
      YY = Matrix::Zero(1, K);
      XY = Matrix::Zero(N, K);
      Dist = Matrix::Zero(N, K);

      XX = X.array().square().rowwise().sum();
      YY = Y.array().square().rowwise().sum().transpose();
      XY = (2.0f * X) * Y.transpose();

      Dist = XX * Matrix::Ones(1, K);
      Dist = Dist + Matrix::Ones(N, 1) * YY;
      Dist = Dist - XY;

      // if there is any negative value, turn it to positive
      // can happen at values very close to zero
      Dist = (Dist.array() < 0.0f).select(-Dist, Dist);

      return Dist;
    }

    /// @brief Main fit function for KInit.
    /// @param X Input data of shape (N, D) -- (number of samples x dimensions)
    /// @param K Desired number of clusters
    /// @param centers Output centers of the clusters
    /// @param indices Output indices for the dataset X corresponding to the cluster centers
    void fit(const Matrix &X, const unsigned int &K,
             Matrix &centers, std::vector<int> &indices)
    {
      // For initial responsibilities, we use the steps 1a -- 1c of the algorithm
      // from: Arthur, D. and Vassilvitskii, S.  "k-means++: the advantages of
      // careful seeding". ACM-SIAM symposium on Discrete algorithms. 2007

      unsigned int N = X.rows();

      // Number of samples should be strictly greater than 1
      if (N <= 1)
      {
        throw std::runtime_error("fit: number of samples is " +
                                 std::to_string(N) +
                                 ", it should be greater than 1.");
      }

      // Number of components should be strictly greater than 1
      if (K <= 1)
      {
        throw std::runtime_error("fit: number of components is " +
                                 std::to_string(K) +
                                 ", it should be greater than 1.");
      }

      // Number of components should be strictly lower than the number of samples
      if (N < K)
      {
        throw std::runtime_error("fit: number of components is " +
                                 std::to_string(K) +
                                 ". It should be strictly smaller than the "
                                 "number of points: " +
                                 std::to_string(N));
      }

      // Set the number of local seeding trials This is what
      // Arthur/Vassilvitskii tried, but did not report specific results for
      // other than mentioning in the conclusion that it helped.
      unsigned int n_local_trials = 2 + std::floor(std::log(K));

      // Pick the first center randomly and track index of point
      int_dist_ = std::uniform_int_distribution<int>(0, N - 1);
      int center_id = int_dist_(generator_);
      centers.row(0) << X.row(center_id);
      indices.push_back(center_id);

      // Initialize list of closest distances and calculate current potential
      Matrix closest_dist_sq = euclideanDistancesSq(centers(0, Eigen::all), X);
      T current_pot = closest_dist_sq.array().sum();

      for (unsigned int i = 1; i < K; i++)
      {
        // choose center candidates by sampling with probability proportional
        // to the squared distance to the closest existing center
        Matrix rand_vals =
            Matrix::Random(n_local_trials, 1).array().abs() * current_pot;
        std::vector<unsigned int> candidate_ids =
            searchSorted(cumSum(closest_dist_sq), rand_vals);

        // compute distances to center candidates
        Matrix distances_to_candidates =
            euclideanDistancesSq(X(candidate_ids, Eigen::all), X);

        // update closest distances squared and potential for each candidate
        for (unsigned int trial = 0; trial < n_local_trials; ++trial)
        {
          Matrix new_dist_sq =
              closest_dist_sq.cwiseMin(distances_to_candidates.row(trial));
          distances_to_candidates.row(trial) << new_dist_sq;
        }
        Matrix candidates_pot = distances_to_candidates.rowwise().sum();

        // decide which candidate is the best
        Eigen::Index best_candidate, temp;
        current_pot = candidates_pot.minCoeff(&best_candidate, &temp);
        closest_dist_sq = distances_to_candidates(best_candidate, Eigen::all);

        // permanently add best center candidate found in local tries
        centers.row(i) << X.row(candidate_ids[best_candidate]);
        indices.push_back(candidate_ids[best_candidate]);
      }
    }

    /// @brief Wrapper over fit function to return tuple of centers and indices
    /// @param X Input data of shape (N, D) -- (number of samples x dimensions)
    /// @param K Desired number of clusters
    /// @return Tuple (centers, indices)
    inline std::pair<Matrix, std::vector<int>> respCalc(const Matrix &X,
                                                        const unsigned int &K)
    {
      // Initialize centers shape is K x D
      Matrix centers = Matrix::Zero(K, D);
      std::vector<int> indices;

      fit(X, K, centers, indices);

      return std::make_pair(centers, indices);
    }

    /// @brief Wrapper over fit function to return the responsibility matrix
    /// @param X Input data of shape (N, D) -- (number of samples x dimensions)
    /// @param resp Output responsibility matrix of shape (N, K) -- (number of samples x number of components)
    inline void getRespMat(const Matrix &X, Matrix &resp)
    {
      unsigned int K = resp.cols();

      // Initialize centers shape is K x D
      Matrix centers = Matrix::Zero(K, D);
      std::vector<int> indices;

      fit(X, K, centers, indices);

      for (unsigned int i = 0; i < K; i++)
      {
        resp(indices[i], i) = 1;
      }
    }

    std::default_random_engine generator_;
    std::uniform_int_distribution<int> int_dist_;
  };
}