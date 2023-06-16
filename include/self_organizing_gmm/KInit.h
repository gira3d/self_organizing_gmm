#pragma once

#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>

#include "TimeProfiler.h"

// to allow for debugging uninitialized matrices
#define EIGEN_INITIALIZE_MATRICES_BY_NAN

// T -- datatypes (usually float or double)
// D -- dimension of the data
template <typename T, uint32_t D>
class KInit
{
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                               (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;

  KInit()
  {
    n_components_ = 1;
  }

  ~KInit()
  {
  }

  KInit(unsigned int n_components)
  {
    n_components_ = n_components;

    initialize(false, "kinit_cpu_stats", "stats.csv");
  }

  KInit(unsigned int n_components, const bool save_stats,
        const std::string& stats_dir, const std::string& stats_file)
  {
    n_components_ = n_components;

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
  }

  inline Matrix cumsum(const Matrix& X)
  {
    Matrix Y = X;

    for (unsigned int i = 1; i < X.cols(); i++)
    {
      Y(Eigen::all, i) += Y(Eigen::all, i - 1);
    }

    return Y;
  }

  inline int ceilSearch(const Matrix& haystack, int low, int high, T needle)
  {
    // get index of ceiling of needle in haystack

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

  inline std::vector<unsigned int> search_sorted(const Matrix& a,
                                                 const Matrix& v)
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

  Matrix euclidean_distances_sq(const Matrix& X, const Matrix& Y)
  {
    tp_.tic("euclideanDist");
    // assumes the input to be (num_samples x dimension)
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
    tp_.toc("euclideanDist");

    return Dist;
  }

  inline std::pair<Matrix, std::vector<int>> resp_calc(const Matrix& X)
  {
    tp_.tic("respCalc");
    // For initial responsibilities, we use the steps 1a -- 1c of the algorithm
    // from: Arthur, D. and Vassilvitskii, S.  "k-means++: the advantages of
    // careful seeding". ACM-SIAM symposium on Discrete algorithms. 2007

    // We assume the input dataset is of shape N x D where
    // N is the number of samples in the dataset
    // D is the dimension of each point in the dataset
    unsigned int n_samples = X.rows();

    // Number of samples should be strictly greater than 1
    if (n_samples <= 1)
    {
      throw std::runtime_error("fit: number of samples is " +
                               std::to_string(n_samples) +
                               ", it should be greater than 1.");
    }

    // Number of components should be strictly greater than 1
    if (n_components_ <= 1)
    {
      throw std::runtime_error("fit: number of components is " +
                               std::to_string(n_components_) +
                               ", it should be greater than 1.");
    }

    // Number of components should be strictly lower than the number of samples
    if (n_samples < n_components_)
    {
      throw std::runtime_error("fit: number of components is " +
                               std::to_string(n_components_) +
                               ". It should be strictly smaller than the "
                               "number of components: " +
                               std::to_string(n_samples));
    }

    // Initialize centers shape is K x D
    Matrix centers = Matrix::Zero(n_components_, D);
    std::vector<int> indices;

    // Set the number of local seeding trials
    // This is what Arthur/Vassilvitskii tried, but did not report
    // specific results for other than mentioning in the conclusion
    // that it helped
    unsigned int n_local_trials = 2 + std::floor(std::log(n_components_));

    // Pick the first center randomly and track index of point
    int_dist_ = std::uniform_int_distribution<int>(0, n_samples - 1);
    int center_id = int_dist_(generator_);
    centers.row(0) << X.row(center_id);
    indices.push_back(center_id);

    // Initialize list of closest distances and calculate current potential
    Matrix closest_dist_sq = euclidean_distances_sq(centers(0, Eigen::all), X);
    T current_pot = closest_dist_sq.array().sum();

    for (unsigned int i = 1; i < n_components_; i++)
    {
      // choose center candidates by sampling with probability proportional
      // to the squared distance to the closest existing center
      Matrix rand_vals =
          Matrix::Random(n_local_trials, 1).array().abs() * current_pot;
      std::vector<unsigned int> candidate_ids =
          search_sorted(cumsum(closest_dist_sq), rand_vals);

      // compute distances to center candidates
      Matrix distances_to_candidates =
          euclidean_distances_sq(X(candidate_ids, Eigen::all), X);

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
    tp_.toc("respCalc");

    return std::make_pair(centers, indices);
  }

  unsigned int n_components_;

  std::default_random_engine generator_;
  std::uniform_int_distribution<int> int_dist_;

  TimeProfiler tp_;
};
