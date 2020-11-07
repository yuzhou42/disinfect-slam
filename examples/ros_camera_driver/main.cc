#include <iostream>
#include <string>
#include <thread>

#include <openvslam/system.h>
#include <popl.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "cameras/l515.h"
#include "cameras/zed_native.h"
#include "utils/time.hpp"
#include "utils/config_reader.hpp"
#include "disinfect_slam/disinfect_slam.h"

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float32MultiArray.h>

using namespace std;

void publishImage(cv_bridge::CvImagePtr & imgBrgPtr, const cv::Mat & img, ros::Publisher & pubImg, std::string imgFrameId, std::string dataType, ros::Time t){
  imgBrgPtr->header.stamp = t;
  imgBrgPtr->header.frame_id = imgFrameId;
  imgBrgPtr->encoding = dataType;
  imgBrgPtr->image = img;
  pubImg.publish(imgBrgPtr->toImageMsg());
}


void run(const ZEDNative &zed_native, const L515 &l515, std::shared_ptr<DISINFSystem> my_sys,  ros::NodeHandle& mNh, tf2_ros::TransformBroadcaster& mTfSlam, bool renderFlag) {
  SE3<float> mSlamPose;
  std::vector<VoxelSpatialTSDF>  mSemanticReconstr;
  std_msgs::Float32MultiArray::Ptr mReconstrMsg(new std_msgs::Float32MultiArray);
  // publishers
  ros::Publisher mPubL515RGB = mNh.advertise<sensor_msgs::Image>("/l515_rgb", 1);
  ros::Publisher mPubL515Depth = mNh.advertise<sensor_msgs::Image>("/l515_depth", 1);
  ros::Publisher mPubZEDImgL = mNh.advertise<sensor_msgs::Image>("/zed_left_rgb", 1);
  ros::Publisher mPubZEDImgR = mNh.advertise<sensor_msgs::Image>("/zed_right_rgb", 1);
  ros::Publisher mPubTsdfGlobal = mNh.advertise<std_msgs::Float32MultiArray>("/tsdf_global", 4);
  ros::Publisher mPubTsdfLocal = mNh.advertise<std_msgs::Float32MultiArray>("/tsdf_local", 4);

  // cv bridges
  cv_bridge::CvImagePtr mL515RGBBrg;
  mL515RGBBrg.reset(new cv_bridge::CvImage);
  cv_bridge::CvImagePtr mL515DepthBrg;
  mL515DepthBrg.reset(new cv_bridge::CvImage);
  cv_bridge::CvImagePtr mZEDImgLBrg;
  mZEDImgLBrg.reset(new cv_bridge::CvImage);
  cv_bridge::CvImagePtr mZEDImgRBrg;
  mZEDImgRBrg.reset(new cv_bridge::CvImage);
  
  // initialize TSDF

  std::thread t_slam([&]() {
    cv::Mat img_left, img_right;
    ros::Time ros_stamp;
    while (ros::ok()) {
      const int64_t timestamp = zed_native.get_stereo_img(&img_left, &img_right);
      my_sys->feed_stereo_frame(img_left, img_right, timestamp);
      ros_stamp.sec = timestamp / 1000;
      ros_stamp.nsec = (timestamp % 1000) * 1000 * 1000;
      publishImage(mZEDImgLBrg, img_left, mPubZEDImgL, "zed", "bgr8" , ros_stamp);
      publishImage(mZEDImgRBrg, img_right, mPubZEDImgR, "zed", "bgr8" , ros_stamp);
      ros::spinOnce();
    }
  });

  std::thread t_tsdf([&]() {
    cv::Mat img_rgb, img_depth;
    ros::Time ros_stamp;
    while (ros::ok()) {
      const int64_t timestamp = l515.get_rgbd_frame(&img_rgb, &img_depth);
      my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp);
      ros_stamp.sec = timestamp / 1000;
      ros_stamp.nsec = (timestamp % 1000) * 1000 * 1000;
      publishImage(mL515RGBBrg, img_rgb, mPubL515RGB, "l515", "rgb8" , ros_stamp);
      publishImage(mL515DepthBrg, img_depth, mPubL515Depth, "l515", "mono16", ros_stamp);
      ros::spinOnce();
    }
  });


  std::thread t_reconst([&]() {
      static unsigned int last_query_time = 0;
      static size_t last_query_amount = 0;
      static float bbox = 4.0;
      static float x_range[2] = {-bbox, bbox};
      static float y_range[2] = {-bbox, bbox};
      static float z_range[2] = {-2.0, 4.0};
      static BoundingCube<float> volumn = {
        x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]};
      static geometry_msgs::TransformStamped transformStamped;
      static ros::Time stamp;
      ros::Rate rate(4);

    while (ros::ok()) {
      uint32_t tsdf_global_sub = mPubTsdfGlobal.getNumSubscribers();
      if(tsdf_global_sub > 0){
        const auto st = get_system_timestamp<std::chrono::milliseconds>();  // nsec
        auto mSemanticReconstr = my_sys->query_tsdf(volumn);
        const auto end = get_system_timestamp<std::chrono::milliseconds>();
        last_query_time = end - st;
        last_query_amount = mSemanticReconstr.size();
        std::cout<<"Last queried %lu voxels "<<last_query_amount<<", took "<< last_query_time<<" ms"<<std::endl;
        int size = last_query_amount * sizeof(VoxelSpatialTSDF);
        mReconstrMsg->data.resize(size);
        std::memcpy(&(mReconstrMsg->data[0]), (char*)mSemanticReconstr.data(),  last_query_amount * sizeof(VoxelSpatialTSDF));
        mPubTsdfGlobal.publish(mReconstrMsg);
      }
      ros::spinOnce();
      rate.sleep();
    }
  });

  std::thread t_pose([&](){
    static ros::Time stamp;
    static ros::Rate rate(30);
    while(ros::ok())
    {
      // u_int64_t tNow = std::chrono::steady_clock::now().time_since_epoch().count();  // nsec
      // stamp.sec += tNow / 1000000000UL; //s
      // stamp.nsec = tNow % 1000000000UL; //ns
      u_int64_t t_query = get_system_timestamp<std::chrono::milliseconds>();
      stamp.sec  = t_query / 1000; //s
      stamp.nsec = (t_query % 1000) * 1000 * 1000; //ns
      SE3<float> mSlamPose = my_sys->query_camera_pose(t_query);
      // pubPose(stamp, mSlamPose, mTfSlam);
      geometry_msgs::TransformStamped transformStamped;
      transformStamped.header.stamp = stamp;
      transformStamped.header.frame_id = "world";
      transformStamped.child_frame_id = "slam";
      transformStamped.transform.translation.x = mSlamPose.m03;
      transformStamped.transform.translation.y = mSlamPose.m13;
      transformStamped.transform.translation.z = mSlamPose.m23;

      // SO3<float> tran(0, 0, -1, 1, 0, 0, 0, 1,0);
      // SO3<float> temp = tran*mSlamPose.GetR();

      // tf2::Matrix3x3 R(temp.m00,temp.m01,temp.m02,
      //                     temp.m10,temp.m11,temp.m12,
      //                     temp.m20,temp.m21,temp.m22);
      tf2::Matrix3x3 R(mSlamPose.m00,mSlamPose.m01,mSlamPose.m02,
                          mSlamPose.m10,mSlamPose.m11,mSlamPose.m12,
                          mSlamPose.m20,mSlamPose.m21,mSlamPose.m22);

      tf2::Quaternion q;
      // q.setRPY(0, 0, msg->theta);
      R.getRotation(q);
      transformStamped.transform.rotation.x = q.x();
      transformStamped.transform.rotation.y = q.y();
      transformStamped.transform.rotation.z = q.z();
      transformStamped.transform.rotation.w = q.w();

      mTfSlam.sendTransform(transformStamped);
      rate.sleep();
    }

  });

  
  if(renderFlag) my_sys->run();
  t_pose.join();
  t_reconst.join();
  t_slam.join();
  t_tsdf.join();
  
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "ros_disinf_slam");
  ros::NodeHandle mNh;
  tf2_ros::TransformBroadcaster mTfSlam;
  std::string model_path, calib_path, orb_vocab_path;
  int devid;
  bool renderFlag;

  mNh.getParam("/ros_disinf_slam/model_path", model_path); 
  mNh.getParam("/ros_disinf_slam/calib_path", calib_path); 
  mNh.getParam("/ros_disinf_slam/orb_vocab_path", orb_vocab_path); 
  mNh.param("/ros_disinf_slam/devid", devid, 2);
  mNh.param("/ros_disinf_slam/renderer", renderFlag, false);
  // ROS_INFO_STREAM(calib_path);

  auto cfg = get_and_set_config(calib_path);
  // initialize cameras
  ZEDNative zed_native(*cfg, devid);
  L515 l515;
  // initialize slam
  std::shared_ptr<DISINFSystem> my_system = std::make_shared<DISINFSystem>(
      calib_path,orb_vocab_path, model_path, renderFlag
  );

  run(zed_native, l515, my_system, mNh, mTfSlam, renderFlag);

  return EXIT_SUCCESS;
}
