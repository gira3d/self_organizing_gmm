#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <math.h>

class CameraModel
{
public:
  CameraModel()
  {
  }
  ~CameraModel()
  {
  }

  CameraModel(const Eigen::Matrix3f& intrinsic_matrix)
  {
    intrinsic_matrix_ = intrinsic_matrix;

    fx_ = intrinsic_matrix_(0, 0);
    fy_ = intrinsic_matrix_(1, 1);
    cx_ = intrinsic_matrix_(0, 2);
    cy_ = intrinsic_matrix_(1, 2);
    im_w_ = static_cast<size_t>(2.0f * cx_ + 0.5f);
    im_h_ = static_cast<size_t>(2.0f * cy_ + 0.5f);
    fx_inv_ = 1.0f / fx_;
    fy_inv_ = 1.0f / fy_;
  }

  CameraModel(const Eigen::Matrix3f& intrinsic_matrix, const size_t& W, const size_t& H)
  {
    intrinsic_matrix_ = intrinsic_matrix;

    fx_ = intrinsic_matrix_(0, 0);
    fy_ = intrinsic_matrix_(1, 1);
    cx_ = intrinsic_matrix_(0, 2);
    cy_ = intrinsic_matrix_(1, 2);
    fx_inv_ = 1.0f / fx_;
    fy_inv_ = 1.0f / fy_;
    im_w_ = W;
    im_h_ = H;
  }

  CameraModel(const float fx, const float fy, const float cx, const float cy,
              const size_t width, const size_t height)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), im_w_(width), im_h_(height)
  {
    fx_inv_ = 1.0f / fx_;
    fy_inv_ = 1.0f / fy_;
    intrinsic_matrix_ << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f;
  }

  std::pair<Eigen::MatrixXf, std::vector<size_t>>
  to_3d(const Eigen::MatrixXf& in, const float& rmin, const float& rmax)
  {
    const size_t W = in.cols();
    const size_t H = in.rows();

    Eigen::MatrixXf ret;
    ret = Eigen::MatrixXf::Zero(W * H, 3);
    std::vector<size_t> valid_indices;
    for (size_t v = 0; v < H; v++)
    {
      for (size_t u = 0; u < W; u++)
      {
        size_t pcl_idx = u + W * v;
        float uf = static_cast<float>(u);
        float vf = static_cast<float>(v);

        if (in(v, u) > rmin && in(v, u) <= rmax)
        {
          valid_indices.push_back(pcl_idx);

          // copy the depth value as it is.
          // assumes that the input is appropriately scaled.
          // z = d
          ret(pcl_idx, 2) = in(v, u);

          // y = (v - cy) * z / fy
          ret(pcl_idx, 1) = (vf - cy_) * ret(pcl_idx, 2) * fy_inv_;

          // x = (u - cx) * z / fx
          ret(pcl_idx, 0) = (uf - cx_) * ret(pcl_idx, 2) * fx_inv_;
        }
      }
    }

    return std::make_pair(ret, valid_indices);
  }

  // The boolean gt is set to true for model evaluation. Otherwise it is kept false.
  std::pair<Eigen::MatrixXf, std::vector<size_t>> to_2d(const Eigen::MatrixXf& in, bool gt)
  {
    return to_2d_dim(in, 2, gt);
  }

  // The boolean gt is set to true for model evaluation. Otherwise it is kept false.
  std::pair<Eigen::MatrixXf, std::vector<size_t>> to_2d_dim(const Eigen::MatrixXf& in, size_t dim, bool gt)
  {
    Eigen::MatrixXf ret = Eigen::MatrixXf::Constant(im_h_, im_w_, static_cast<float>(-1.0));
    Eigen::MatrixXf depth_im = Eigen::MatrixXf::Constant(im_h_, im_w_, static_cast<float>(-1.0));

    std::vector<size_t> invalid_indices;

    float uf, vf, inv_z;
    for (size_t i = 0; i < in.rows(); i++)
    {
      inv_z = 1.0f / in(i, 2);
      uf = (fx_ * in(i, 0)) * inv_z + cx_ + 0.5f;
      vf = (fy_ * in(i, 1)) * inv_z + cy_ + 0.5f;

      size_t u = static_cast<size_t>(floor(uf));
      size_t v = static_cast<size_t>(floor(vf));

      if (u >= 0 && u < im_w_ && v >= 0 && v < im_h_)
      {
        if (ret(v, u) < 0.0 || depth_im(v, u) < 0.0)
        {
          ret(v, u) = in(i, dim);
          depth_im(v, u) = in(i, 2);
        }
        else
        {
          if (in(i, 2) < depth_im(v, u))
          {
            ret(v, u) = in(i, dim);
            depth_im(v, u) = in(i, 2);
          }
          else
          {
            if (gt)
            {
              invalid_indices.push_back(i);
            }
          }
        }
      }
      else
      {
        invalid_indices.push_back(i);
      }
    }

    return std::make_pair(ret, invalid_indices);

  }


  size_t im_w_, im_h_;
  float fx_, fy_, cx_, cy_;
  float fx_inv_, fy_inv_;
  Eigen::Matrix3f intrinsic_matrix_;
};
