#include <yaml-cpp/yaml.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <popl.hpp>
#include <sl/Camera.hpp>
#include <string>
#include <thread>

#include "cameras/zed.h"
#include "utils/time.hpp"
#include "System.h"
#define NANO2SEC(ns) (double(ns) / 1e9)

void tracking(const std::string& vocab_file_path, const std::string& config_file_path,
              const std::string& logdir, ZED* camera) {
  ORB_SLAM3::System SLAM(vocab_file_path, config_file_path,
                         ORB_SLAM3::System::IMU_STEREO, true);
  std::thread th_imu([&]() {
    // const auto start = std::chrono::steady_clock::now();
    while (true) {
      camera->GetIMU();
    }
  });

  std::thread th_img([&]() {
    // const auto start = std::chrono::steady_clock::now();
    cv::Mat img_left, img_right;
    cv::Mat M1l,M2l,M1r,M2r;
    int64_t t_img;
    doRectify(config_file_path, &M1l, &M1r, &M2l, &M2r);
    while (true) {
      t_img = camera->GetStereoFrame(&img_left, &img_right);
      // cv::imshow("left_image", img_left);
      // cv::waitKey(1);
      if(t_img > camera->qImuBuf.back().t) continue;
      
      vector<ORB_SLAM3::IMU::Point> vImuMeas;
      camera->mBufMutex.lock();
      if(!camera->qImuBuf.empty())
      {
        // Load imu measurements from buffer
        vImuMeas.clear();
        while(!camera->qImuBuf.empty() && camera->qImuBuf.front().t<=t_img)
        {
          double t = NANO2SEC(camera->qImuBuf.front().t);
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(camera->qImuBuf.front().a,camera->qImuBuf.front().w,t));
          camera->qImuBuf.pop();
        }
      }
      camera->mBufMutex.unlock();

      cv::remap(img_left,img_left,M1l,M2l,cv::INTER_LINEAR);
      cv::remap(img_right,img_right,M1r,M2r,cv::INTER_LINEAR);
      cv::Mat Tcw = SLAM.TrackStereo(img_left,img_right,NANO2SEC(t_img),vImuMeas);
    }
  });
  th_imu.join();
  th_img.join();
  SLAM.Shutdown();
}

int main(int argc, char* argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
  auto log_dir =
      op.add<popl::Value<std::string>>("", "logdir", "directory to store logged data", "./log");
  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (help->is_set()) {
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (!vocab_file_path->is_set() || !config_file_path->is_set()) {
    std::cerr << "Invalid Arguments" << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  ZED camera;
  ORB_SLAM3::Verbose::th = ORB_SLAM3::Verbose::VERBOSITY_DEBUG;
  tracking( vocab_file_path->value(), config_file_path->value(), log_dir->value(), &camera);

  return EXIT_SUCCESS;
}
