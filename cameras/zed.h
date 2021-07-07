#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <sl/Camera.hpp>
#include "utils/time.hpp"
#include<queue>
#include<thread>
#include<mutex>
/**
 * @brief ZED camera interface using ZED SDK
 */
struct simuData{
  cv::Vec4d q;
  cv::Point3f a;
  cv::Point3f w;
  int64_t t;
};

void doRectify(const std::string& config_file_path, cv::Mat* M1l, cv::Mat* M1r, cv::Mat* M2l, cv::Mat* M2r);
class ZED {
 public:
  ZED();
  ~ZED();

  /**
   * @return camera config including image specs and calibration parameters
   */
  sl::CameraConfiguration GetConfig() const;

  /**
   * @brief read both stereo and RGBD frames
   *
   * @param left_img    left image of stereo frame
   * @param right_img   right image of stereo frame
   * @param rgb_img     rgb image of RGBD frame
   * @param depth_img   depth image of RGBD frame
   */
  double GetStereoFrame(cv::Mat* left_img, cv::Mat* right_img);
  void GetStereoAndRGBDFrame(cv::Mat* left_img, cv::Mat* right_img, cv::Mat* rgb_img,
                             cv::Mat* depth_img);
  void GetIMU();

  sl::Camera zed_;
  sl::SensorsData sensors_data_;
  sl::CameraConfiguration config_;
  sl::RuntimeParameters rt_params_;
  std::queue<simuData> qImuBuf;  // IMU buffer 
  std::mutex mBufMutex;
 private:
  void AllocateIfNeeded(cv::Mat* img, int type) const;
};
