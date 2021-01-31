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
#include <image_transport/image_transport.h>


using namespace std;
// std::mutex mtx;   
cv::Mat zedLeftMask;
cv::Mat l515Mask;

void publishImage(cv_bridge::CvImagePtr & imgBrgPtr, const cv::Mat & img, ros::Publisher & pubImg, std::string imgFrameId, std::string dataType, ros::Time t){
  imgBrgPtr->header.stamp = t;
  imgBrgPtr->header.frame_id = imgFrameId;
  imgBrgPtr->encoding = dataType;
  imgBrgPtr->image = img;
  pubImg.publish(imgBrgPtr->toImageMsg());
}


void run(const ZEDNative &zed_native, const L515 &l515, std::shared_ptr<DISINFSystem> my_sys,  ros::NodeHandle& mNh, tf2_ros::TransformBroadcaster& mTfSlam, bool renderFlag, double   bbox_xy) {
  SE3<float> mSlamPose;
  std::vector<VoxelSpatialTSDF>  mSemanticReconstr;
  std_msgs::Float32MultiArray::Ptr mReconstrMsg(new std_msgs::Float32MultiArray);
  geometry_msgs::TransformStamped transformStamped;

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
    cv::Mat img_left, img_right, zedLeftMaskL;
    ros::Time ros_stamp;
    while (ros::ok()) {
      const int64_t timestamp = zed_native.get_stereo_img(&img_left, &img_right);
      zedLeftMaskL = zedLeftMask.clone();
      my_sys->feed_stereo_frame(img_left, img_right, timestamp, zedLeftMaskL);
      ros_stamp.sec = timestamp / 1000;
      ros_stamp.nsec = (timestamp % 1000) * 1000 * 1000;
      if(mPubZEDImgL.getNumSubscribers()>0)
        publishImage(mZEDImgLBrg, img_left, mPubZEDImgL, "zed", "bgr8" , ros_stamp);
      if(mPubZEDImgR.getNumSubscribers()>0)
        publishImage(mZEDImgRBrg, img_right, mPubZEDImgR, "zed", "bgr8" , ros_stamp);
      ros::spinOnce();
    }
  });

  std::thread t_tsdf([&]() {
    cv::Mat img_rgb, img_depth, l515MaskL;
    ros::Time ros_stamp;
    while (ros::ok()) {
      const int64_t timestamp = l515.get_rgbd_frame(&img_rgb, &img_depth);
      l515MaskL = l515Mask.clone();
      my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp,l515MaskL);
      ros_stamp.sec = timestamp / 1000;
      ros_stamp.nsec = (timestamp % 1000) * 1000 * 1000;
      if(mPubL515RGB.getNumSubscribers()>0)
        publishImage(mL515RGBBrg, img_rgb, mPubL515RGB, "l515", "rgb8" , ros_stamp);
      if(mPubL515Depth.getNumSubscribers()>0)
        publishImage(mL515DepthBrg, img_depth, mPubL515Depth, "l515", "mono16", ros_stamp);
      ros::spinOnce();
    }
  });


  std::thread t_reconst([&]() {
      static unsigned int last_query_time = 0;
      static size_t last_query_amount = 0;
      // static float bbox = 4.0;
      static float x_range[2] = {-bbox_xy, bbox_xy};
      static float y_range[2] = {-bbox_xy, bbox_xy};
      static float z_range[2] = {-2.0, 4.0};
      static BoundingCube<float> volumn = {
        x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]};
      static ros::Time stamp;
      ros::Rate rate(4);

    while (ros::ok()) {
      uint32_t tsdf_global_sub = mPubTsdfGlobal.getNumSubscribers();
      uint32_t tsdf_local_sub = mPubTsdfLocal.getNumSubscribers();

      if(tsdf_global_sub > 0){
        const auto st = get_timestamp<std::chrono::milliseconds>();  // nsec
        auto mSemanticReconstr = my_sys->query_tsdf(volumn);
        const auto end = get_timestamp<std::chrono::milliseconds>();
        last_query_time = end - st;
        last_query_amount = mSemanticReconstr.size();
        std::cout<<"Last queried %lu voxels "<<last_query_amount<<", took "<< last_query_time<<" ms"<<std::endl;
        int size = last_query_amount * sizeof(VoxelSpatialTSDF);
        mReconstrMsg->data.resize(size);
        std::memcpy(&(mReconstrMsg->data[0]), (char*)mSemanticReconstr.data(),  last_query_amount * sizeof(VoxelSpatialTSDF));
        mPubTsdfGlobal.publish(mReconstrMsg);
      }

      if(tsdf_local_sub > 0){
        const auto st = get_timestamp<std::chrono::milliseconds>();  // nsec
        float x_off = transformStamped.transform.translation.x, y_off = transformStamped.transform.translation.y, z_off = transformStamped.transform.translation.z;
        std::cout<<"x_off: "<<x_off<<"  y_off: "<<y_off<<"  z_off: "<<z_off<<std::endl; 
        BoundingCube<float> volumn_local = {
        x_off + x_range[0], x_off+ x_range[1], y_off + y_range[0], y_off + y_range[1], z_off + z_range[0], z_off + z_range[1]};
        auto mSemanticReconstr = my_sys->query_tsdf(volumn_local);
        const auto end = get_timestamp<std::chrono::milliseconds>();
        last_query_time = end - st;
        last_query_amount = mSemanticReconstr.size();
        std::cout<<"Last queried %lu voxels "<<last_query_amount<<", took "<< last_query_time<<" ms"<<std::endl;
        int size = last_query_amount * sizeof(VoxelSpatialTSDF);
        mReconstrMsg->data.resize(size);
        std::memcpy(&(mReconstrMsg->data[0]), (char*)mSemanticReconstr.data(),  last_query_amount * sizeof(VoxelSpatialTSDF));
        mPubTsdfLocal.publish(mReconstrMsg);
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
      u_int64_t t_query = get_timestamp<std::chrono::milliseconds>();
      stamp.sec  = t_query / 1000; //s
      stamp.nsec = (t_query % 1000) * 1000 * 1000; //ns
      mSlamPose = my_sys->query_camera_pose(t_query);
      // pubPose(stamp, mSlamPose, mTfSlam);
      tf2::Transform tf2_trans;
      tf2::Transform tf2_trans_inv;

      tf2::Matrix3x3 R(mSlamPose.m00,mSlamPose.m01,mSlamPose.m02,
                          mSlamPose.m10,mSlamPose.m11,mSlamPose.m12,
                          mSlamPose.m20,mSlamPose.m21,mSlamPose.m22);
      tf2::Vector3 T(mSlamPose.m03, mSlamPose.m13, mSlamPose.m23);

      tf2_trans.setBasis(R);
      tf2_trans.setOrigin(T);

      tf2_trans_inv = tf2_trans.inverse();

      transformStamped.header.stamp = stamp;
      transformStamped.header.frame_id = "slam";
      transformStamped.child_frame_id = "zed";

      tf2::Quaternion q = tf2_trans_inv.getRotation();
      transformStamped.transform.rotation.x = q.x();
      transformStamped.transform.rotation.y = q.y();
      transformStamped.transform.rotation.z = q.z();
      transformStamped.transform.rotation.w = q.w();

      tf2::Vector3 t = tf2_trans_inv.getOrigin();
      transformStamped.transform.translation.x = t[0];
      transformStamped.transform.translation.y = t[1];
      transformStamped.transform.translation.z = t[2];
      
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

void zedMaskCb(const sensor_msgs::ImageConstPtr& msg)
{
  zedLeftMask = cv_bridge::toCvShare(msg, "8UC1")->image;
  // try
  // {
  //   cv::imshow("zedMask", cv_bridge::toCvShare(msg, "bgr8")->image);
  //   cv::waitKey(30);
  // }
  // catch (cv_bridge::Exception& e)
  // {
  //   ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  // }
}

void l515MaskCb(const sensor_msgs::ImageConstPtr& msg)
{
  // image encoding: http://docs.ros.org/en/jade/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html
  // l515Mask = cv_bridge::toCvShare(msg,  "bgr8")->image;
  // l515Mask.convertTo(l515Mask, CV_8UC1); // depth scale
  
  l515Mask = cv_bridge::toCvShare(msg,  "8UC1")->image;

  // try
  // {
  //   cv::imshow("l515Mask", cv_bridge::toCvShare(msg, "bgr8")->image);
  //   cv::waitKey(30);
  // }
  // catch (cv_bridge::Exception& e)
  // {
  //   ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  // }
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "ros_disinf_slam");
  ros::NodeHandle mNh;

  image_transport::ImageTransport it(mNh);
  image_transport::Subscriber sub1 = it.subscribe("/zed2/zed_node/right/image_rect_color", 1, zedMaskCb);
  image_transport::Subscriber sub2 = it.subscribe("/zed2/zed_node/left/image_rect_color", 1, l515MaskCb);

  tf2_ros::TransformBroadcaster mTfSlam;
  std::string model_path, calib_path, orb_vocab_path;
  int devid;
  bool renderFlag;
  double bbox_xy;

  mNh.getParam("/ros_disinf_slam/model_path", model_path); 
  mNh.getParam("/ros_disinf_slam/calib_path", calib_path); 
  mNh.getParam("/ros_disinf_slam/orb_vocab_path", orb_vocab_path); 
  mNh.param("/ros_disinf_slam/devid", devid, 2);
  mNh.param("/ros_disinf_slam/bbox_xy", bbox_xy, 4.0);
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

  run(zed_native, l515, my_system, mNh, mTfSlam, renderFlag, bbox_xy);

  return EXIT_SUCCESS;
}
