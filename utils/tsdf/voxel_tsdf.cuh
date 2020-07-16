#pragma once

#include <opencv2/opencv.hpp>

#include "utils/cuda/lie_group.cuh"
#include "utils/cuda/camera.cuh"
#include "utils/tsdf/voxel_hash.cuh"

class TSDFGrid {
 public:
  TSDFGrid(float voxel_size, float truncation, float max_depth);
  ~TSDFGrid();

  void Integrate(const cv::Mat &img_rgb, const cv::Mat &img_depth, 
                 const CameraIntrinsics<float> &intrinsics, const SE3<float> &cam_P_world);

  void RayCast(cv::Mat *img_normal, const CameraParams &virtual_cam_params, 
                                    const SE3<float> &cam_P_world);

 protected:
  cudaStream_t stream_;
  // voxel grid params
  VoxelHashTable hash_table_;
  const float voxel_size_;
  const float truncation_;
  const float max_depth_;

  // voxel data buffer
  VoxelBlock *visible_blocks_;
  int *visible_mask_;
  int *visible_indics_;
  int *visible_indics_aux_;
  // image data buffer
  uchar3 *img_rgb_;
  float *img_depth_;
  float *img_depth_to_range_;
};

