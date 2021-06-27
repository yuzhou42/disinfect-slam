#include "zed.h"
#define DEG2RAD 0.017453293
#define RAD2DEG 57.295777937

ZED::ZED() {
  sl::InitParameters init_params;
  // No depth computation required here
  init_params.depth_mode = sl::DEPTH_MODE::NONE;
  init_params.camera_resolution = sl::RESOLUTION::VGA;
  init_params.camera_fps = 30;
  init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;  // Yu: Important to set!!!
  init_params.coordinate_units = sl::UNIT::METER;
  init_params.enable_image_enhancement = true; // Always active

  // init_params.depth_mode = sl::DEPTH_MODE::QUALITY;
  // Open the camera
  sl::ERROR_CODE err = zed_.open(init_params);
  if (err != sl::ERROR_CODE::SUCCESS) {
      std::cout << "Error " << err << ", exit program.\n";
  }

  zed_.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, 100);
  rt_params_ = zed_.getRuntimeParameters();
  rt_params_.confidence_threshold = 50;
  config_ = zed_.getCameraInformation().camera_configuration;
}

ZED::~ZED() { zed_.close(); }

sl::CameraConfiguration ZED::GetConfig() const { return config_; }

// Todo: create a timer for obtaining IMU data
void ZED::GetIMU(){
  static int64_t tz_prev_imu = 0;
  static int cnt = 0;
  static int64_t total_t = 0;
  if(zed_.getSensorsData(sensors_data_, sl::TIME_REFERENCE::CURRENT) != sl::ERROR_CODE::SUCCESS ) {
            std::cout<<"Not retrieved sensors data in CURRENT REFERENCE TIME"<<std::endl;
            return;
        }
  //Incoming packets are time stamped at reception by the host machine, in Epoch time and nanosecond resolution
  int64_t tz_imu = sensors_data_.imu.timestamp;
  if(tz_imu == tz_prev_imu) return;
  cnt++;
  total_t += (tz_imu - tz_prev_imu);
  if (total_t>=1e9){
    std::cout<<"freq of imu: "<<cnt<<std::endl; //405
    cnt = 0;
    total_t = 0;
  }
  tz_prev_imu = tz_imu;
  cv::Vec4d q(sensors_data_.imu.pose.getOrientation()[0],
              sensors_data_.imu.pose.getOrientation()[1],
              sensors_data_.imu.pose.getOrientation()[2],
              sensors_data_.imu.pose.getOrientation()[3]); // x,y,z,w
  cv::Point3f gyr(sensors_data_.imu.angular_velocity[0] * DEG2RAD,sensors_data_.imu.angular_velocity[1] * DEG2RAD,sensors_data_.imu.angular_velocity[2] * DEG2RAD);
  cv::Point3f acc(sensors_data_.imu.linear_acceleration[0],sensors_data_.imu.linear_acceleration[1],sensors_data_.imu.linear_acceleration[2]);
  
  simuData imuData;
  imuData.w = gyr;
  imuData.a = acc;
  imuData.t = tz_imu;
  imuData.q = q;
  // std::cout<<"before lock in driver"<<std::endl;
  // mBufMutex.lock();
  qImuBuf.push(std::move(imuData)); //double check
  // mBufMutex.lock();

    // sensors_data.imu contains new data
  // std::cout << "IMU Orientation: {" << sensors_data_.imu.pose.getOrientation() << "}"<<std::endl;
  // std::cout << "IMU Linear Acceleration: {" << sensors_data_.imu.linear_acceleration << "} [m/sec^2]"<<std::endl;
  // std::cout << "IMU Angular Velocity: {" << sensors_data_.imu.angular_velocity << "} [deg/sec]"<<std::endl;
  // std::cout <<"IMU time: "<<tz_imu/1e9<<std::endl;
}

double ZED::GetStereoFrame(cv::Mat* left_img, cv::Mat* right_img) {
  static int64_t tz_prev_img = 0;
  static int cnt = 0;
  static int64_t total_t = 0;
  cnt++;
  AllocateIfNeeded(left_img, CV_8UC1);
  AllocateIfNeeded(right_img, CV_8UC1);
  sl::Mat left_sl(config_.resolution, sl::MAT_TYPE::U8_C1, left_img->data,
                  config_.resolution.width);
  sl::Mat right_sl(config_.resolution, sl::MAT_TYPE::U8_C1, right_img->data,
                   config_.resolution.width);

  if (zed_.grab(rt_params_) == sl::ERROR_CODE::SUCCESS) {
    zed_.retrieveImage(left_sl, sl::VIEW::LEFT_GRAY);
    zed_.retrieveImage(right_sl, sl::VIEW::RIGHT_GRAY);
  }
  int64_t tz_img = zed_.getTimestamp(sl::TIME_REFERENCE::IMAGE); // Get image timestamp

  //the system time is 20075950 nano seconds slower than the time from the sensor driver
  // const auto ts_img = (int64_t)(GetSystemTimestamp<std::chrono::nanoseconds>());
  // std::cout<<"time diff: "<<ts_img - tz_img<<std::endl;  

  total_t += (tz_img - tz_prev_img);
  if (total_t >= 1e9){
    std::cout<<"freq of image: "<<cnt<<std::endl; //30
    cnt = 0;
    total_t = 0;
  }
  tz_prev_img = tz_img;
  return tz_img;
}


void ZED::GetStereoAndRGBDFrame(cv::Mat* left_img, cv::Mat* right_img, cv::Mat* rgb_img,
                                cv::Mat* depth_img) {
  AllocateIfNeeded(left_img, CV_8UC1);
  AllocateIfNeeded(right_img, CV_8UC1);
  AllocateIfNeeded(rgb_img, CV_8UC4);
  AllocateIfNeeded(depth_img, CV_32FC1);

  sl::Mat left_sl(config_.resolution, sl::MAT_TYPE::U8_C1, left_img->data,
                  config_.resolution.width);
  sl::Mat right_sl(config_.resolution, sl::MAT_TYPE::U8_C1, right_img->data,
                   config_.resolution.width);
  sl::Mat rgb_sl(config_.resolution, sl::MAT_TYPE::U8_C4, rgb_img->data,
                 config_.resolution.width * 4);
  sl::Mat depth_sl(config_.resolution, sl::MAT_TYPE::F32_C1, depth_img->data,
                   config_.resolution.width * sizeof(float));

  if (zed_.grab(rt_params_) == sl::ERROR_CODE::SUCCESS) {
    zed_.retrieveImage(left_sl, sl::VIEW::LEFT_GRAY);
    zed_.retrieveImage(right_sl, sl::VIEW::RIGHT_GRAY);
    zed_.retrieveImage(rgb_sl, sl::VIEW::LEFT);
    zed_.retrieveMeasure(depth_sl, sl::MEASURE::DEPTH);
  }
}

void ZED::AllocateIfNeeded(cv::Mat* img, int type) const {
  if (img->empty() || img->type() != type || img->cols != config_.resolution.width ||
      img->rows != config_.resolution.height)
    *img = cv::Mat(config_.resolution.height, config_.resolution.width, type);
}
