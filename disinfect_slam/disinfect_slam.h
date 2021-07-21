#pragma once

// #include <openvslam/publish/map_publisher.h>
// #include <openvslam/system.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <popl.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// #include "modules/renderer_module.h"
// #include "modules/slam_module.h"
#include "modules/tsdf_module.h"
// #include "segmentation/inference.h"
#include "utils/config_reader.hpp"
#include "utils/cuda/errors.cuh"
// #include "utils/gl/renderer_base.h"
#include "utils/rotation_math/pose_manager.h"
#include "utils/time.hpp"
#include "System.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#define NANO2SEC(ns) (double(ns) / 1e9)

class DISINFSystem {
 public:
  DISINFSystem(std::string camera_config_path, std::string vocab_path, std::string seg_model_path, 
                const ORB_SLAM3::System::eSensor sensor, bool pangolin_view = false);
  ~DISINFSystem();

  void feed_rgbd_frame(const cv::Mat& img_rgb, const cv::Mat& img_depth, int64_t timestamp, const cv::Mat& mask = cv::Mat{});

  void feed_stereo(const cv::Mat& img_left, const cv::Mat& img_right, double timestamp);

  void feed_stereo_IMU(const cv::Mat& img_left, const cv::Mat& img_right, double timestamp, const std::vector<ORB_SLAM3::IMU::Point>& vImuMeas);

  std::vector<VoxelSpatialTSDF> query_tsdf(const BoundingCube<float>& volumn);

  SE3<float> query_camera_pose(const int64_t timestamp);

  void run();

 private:
  std::shared_ptr<ORB_SLAM3::System> SLAM_;
  std::shared_ptr<TSDFSystem> TSDF_;
  std::shared_ptr<pose_manager> camera_pose_manager_;
  // std::shared_ptr<inference_engine> SEG_;
  // std::shared_ptr<ImageRenderer> RENDERER_;
  float depthmap_factor_;
};
