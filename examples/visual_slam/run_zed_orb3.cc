#include <yaml-cpp/yaml.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <popl.hpp>
#include <sl/Camera.hpp>
#include <string>
#include <thread>

#include "cameras/zed.h"
#include "utils/time.hpp"
#include "System.h"
#define NANO2SEC(ns) (double(ns) / 1e9)

int doRectify(const std::string& config_file_path, cv::Mat* M1l, cv::Mat* M1r, cv::Mat* M2l, cv::Mat* M2r){
  std::cout<<"~~~~~~~do rectify~~~~~~~~~~"<<std::endl;
  // Load settings related to stereo calibration
  cv::FileStorage fsSettings(config_file_path, cv::FileStorage::READ);
  if(!fsSettings.isOpened())
  {
      cerr << "ERROR: Wrong path to settings" << endl;
      return -1;
  }

  cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r, T10;
  fsSettings["LEFT.K"] >> K_l;
  fsSettings["RIGHT.K"] >> K_r;

  fsSettings["LEFT.P"] >> P_l;
  fsSettings["RIGHT.P"] >> P_r;                       

  fsSettings["LEFT.R"] >> R_l;
  fsSettings["RIGHT.R"] >> R_r;

  fsSettings["LEFT.D"] >> D_l;
  fsSettings["RIGHT.D"] >> D_r;

  // add stereoRectify to get R1, R2, P1, P2
  fsSettings["T10"] >> T10;

  int rows_l = fsSettings["LEFT.height"];
  int cols_l = fsSettings["LEFT.width"];
  int rows_r = fsSettings["RIGHT.height"];
  int cols_r = fsSettings["RIGHT.width"];

  if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
          rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
  {
      cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
      return -1;
  }
  cv::Mat R1, R2, P1, P2, Q;
  cv::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(cols_l,rows_l),T10.rowRange(0,3).colRange(0,3),T10.rowRange(0,3).colRange(3,4),R1,R2,P1,P2,Q);
  std::cout<<"R: " << endl<<T10.rowRange(0,3).colRange(0,3)<<std::endl;
  std::cout<<"T: " << endl<<T10.rowRange(0,3).colRange(3,4)<<std::endl;
  
  std::cout<<"R1: " << endl<<R1<<std::endl;
  std::cout<<"R2: " << endl<<R2<<std::endl;
  std::cout<<"P1: " << endl<<P1<<std::endl;
  std::cout<<"P2: " << endl<<P2<<std::endl;
  cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,*M1l,*M2l);
  cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,*M1r,*M2r);
}

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
