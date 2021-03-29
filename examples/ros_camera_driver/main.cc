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

#include <KrisLibrary/geometry/TSDFReconstruction.h>
#include <KrisLibrary/math3d/AABB3D.h>
#include <KrisLibrary/meshing/IO.h>
#include <KrisLibrary/meshing/TriMeshOperators.h>
#include <KrisLibrary/utils/ioutils.h>
#include <KrisLibrary/utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <ros/ros.h>
#include <shape_msgs/Mesh.h>
#include <geometry_msgs/Point.h>
#include <shape_msgs/MeshTriangle.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2/convert.h>
#include <chrono>
#include <cinttypes>

#define CELL_SIZE 0.05
#define TRUNCATION_DISTANCE -0.1

using namespace std;
// std::mutex mtx;   
cv::Mat zedLeftMask;
cv::Mat l515Mask;
std::mutex mask_lock;
std::mutex zed_mask_lock;

rviz_visual_tools::RvizVisualToolsPtr visual_tools_;
tf2_ros::Buffer tfBuffer;
ros::Publisher meshPub;
std::shared_ptr<tf2_ros::TransformListener> tfListener;
std_msgs::Float32MultiArray::Ptr mReconstrMsg;
geometry_msgs::TransformStamped transformStamped;

void publishImage(cv_bridge::CvImagePtr & imgBrgPtr, const cv::Mat & img, ros::Publisher & pubImg, std::string imgFrameId, std::string dataType, ros::Time t){
  imgBrgPtr->header.stamp = t;
  imgBrgPtr->header.frame_id = imgFrameId;
  imgBrgPtr->encoding = dataType;
  imgBrgPtr->image = img;
  pubImg.publish(imgBrgPtr->toImageMsg());
}

void tsdfCb(const std_msgs::Float32MultiArray::Ptr& msg)
{
    static bool init = false;
    static Eigen::Isometry3d pose;
    static geometry_msgs::TransformStamped transformStampedInit;
    if(!init){
        try{
        transformStampedInit = tfBuffer.lookupTransform("world", "slam",
                                ros::Time(0));
        pose = tf2::transformToEigen(transformStampedInit);

        }
        catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());
        }
        init = true;
    }

    auto values = msg->data;
    // ROS_INFO("I heard tsdf of size: ", msg->data.size());
    const auto st = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());

    int numPoints = msg->data.size()/4;
    float minValue = 1e100, maxValue = -1e100;
    Math3D::AABB3D bbox;
    int k=0;
    for(int i=0;i<numPoints;i++,k+=4) {
        minValue = Min(minValue,values[k+3]);
        maxValue = Max(maxValue,values[k+3]);
        bbox.expand(Math3D::Vector3(values[k],values[k+1],values[k+2]));
    }
    // printf("Read %d points with distance in range [%g,%g]\n",numPoints,minValue,maxValue);
    // printf("   x range [%g,%g]\n",bbox.bmin.x,bbox.bmax.x);
    // printf("   y range [%g,%g]\n",bbox.bmin.y,bbox.bmax.y);
    // printf("   z range [%g,%g]\n",bbox.bmin.z,bbox.bmax.z);
    float truncation_distance = TRUNCATION_DISTANCE;
    if(TRUNCATION_DISTANCE < 0) {
        //auto-detect truncation distance
        truncation_distance = Max(-minValue,maxValue)*0.99;
        printf("Auto-detected truncation distance %g\n",truncation_distance);
    }
    // printf("Using cell size %g\n",CELL_SIZE);
    Geometry::SparseTSDFReconstruction tsdf(Math3D::Vector3(CELL_SIZE),truncation_distance);
    tsdf.tsdf.defaultValue[0] = truncation_distance;
    k=0;
    Math3D::Vector3 ofs(CELL_SIZE*0.5);
    for(int i=0;i<numPoints;i++,k+=4) {
        tsdf.tsdf.SetValue(Math3D::Vector3(values[k],values[k+1],values[k+2])+ofs,values[k+3]);
    }

    // printf("Extracting mesh\n");
    Meshing::TriMesh mesh;
    tsdf.ExtractMesh(mesh);
    // std::cout<<"Before Merge: vertsSize: "<<mesh.verts.size()<<std::endl;
    // std::cout<<"Before Merge: trisSize: "<<mesh.tris.size()<<std::endl;

    MergeVertices(mesh, 0.05);

    int vertsSize = mesh.verts.size();
    int trisSize = mesh.tris.size();

    std::cout<<"vertsSize: "<<vertsSize<<std::endl;
    std::cout<<"trisSize: "<<trisSize<<std::endl;
    const auto end = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
    std::cout<<"mesh processing time: "<<end-st<<" ms"<<std::endl;

    const auto st_msg = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());  
    shape_msgs::Mesh::Ptr mMeshMsg = boost::make_shared<shape_msgs::Mesh>();
    // geometry_msgs/Point[] 
    mMeshMsg->vertices.resize(vertsSize);

    // // // shape_msgs/MeshTriangle[] 
    mMeshMsg->triangles.resize(trisSize);
    

    for(int i = 0; i < vertsSize; i++){
        mMeshMsg->vertices[i].x = mesh.verts[i].x;
        mMeshMsg->vertices[i].y = mesh.verts[i].y;
        mMeshMsg->vertices[i].z = mesh.verts[i].z;
        // std::cout<<mesh.verts[i].x<<std::endl;
    }

    for(int i = 0; i < trisSize; i++){
        mMeshMsg->triangles[i].vertex_indices[0] = mesh.tris[i].a;
        mMeshMsg->triangles[i].vertex_indices[1] = mesh.tris[i].b;
        mMeshMsg->triangles[i].vertex_indices[2] = mesh.tris[i].c;
        // std::cout<<mesh.tris[i].a<<std::endl;
    }
    meshPub.publish(mMeshMsg);
    // Eigen::Isometry3d pose;
    // pose = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()); // rotate along X axis by 45 degrees
    // pose.translation() = Eigen::Vector3d( 0, 0, 0 ); // translate x,y,z
    // Publish arrow vector of pose
    visual_tools_->publishMesh(pose, *mMeshMsg, rviz_visual_tools::ORANGE, 1, "mesh", 1); // rviz_visual_tools::TRANSLUCENT_LIGHT

    const auto end_msg = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());  
    // Don't forget to trigger the publisher!
    visual_tools_->trigger();
}

void run(const ZEDNative &zed_native, const L515 &l515, std::shared_ptr<DISINFSystem> my_sys,  ros::NodeHandle& mNh, tf2_ros::TransformBroadcaster& mTfSlam, bool renderFlag, double   bbox_xy, bool global_mesh) {
  std::vector<VoxelSpatialTSDF>  mSemanticReconstr;

  // publishers
  ros::Publisher mPubL515RGB = mNh.advertise<sensor_msgs::Image>("/l515_rgb", 1);
  ros::Publisher mPubL515Depth = mNh.advertise<sensor_msgs::Image>("/l515_depth", 1);
  ros::Publisher mPubZEDImgL = mNh.advertise<sensor_msgs::Image>("/zed_left_rgb", 1);
  ros::Publisher mPubZEDImgR = mNh.advertise<sensor_msgs::Image>("/zed_right_rgb", 1);
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
      const int64_t timestamp = zed_native.GetStereoFrame(&img_left, &img_right);
      zed_mask_lock.lock();
      zedLeftMaskL = zedLeftMask.clone();
      zed_mask_lock.unlock();
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
      const int64_t timestamp = l515.GetRGBDFrame(&img_rgb, &img_depth);
        mask_lock.lock();
        l515MaskL = l515Mask.clone();
        mask_lock.unlock();
      // my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp,l515MaskL);
      my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp);

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
      static float z_range[2]           = {0, 8};
      static BoundingCube<float> volumn = {
        x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]};
      static ros::Time stamp;
      ros::Rate rate(4);

    while (ros::ok()) {
        if (!global_mesh)
      {
          const auto st = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
          float x_off   = transformStamped.transform.translation.x,
                y_off   = transformStamped.transform.translation.y,
                z_off   = transformStamped.transform.translation.z;
          std::cout << "x_off: " << x_off << "  y_off: " << y_off << "  z_off: " << z_off
                    << std::endl;
          volumn = {x_off + x_range[0],
                      x_off + x_range[1],
                      y_off + y_range[0],
                      y_off + y_range[1],
                      z_off + z_range[0],
                      z_off + z_range[1]};
      }

      const auto st          = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
      auto mSemanticReconstr           = my_sys->query_tsdf(volumn);
      const auto end                   = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
      last_query_time                  = end - st;
      last_query_amount                = mSemanticReconstr.size();
      std::cout << "Last queried %lu voxels " << last_query_amount << ", took " << last_query_time
                  << " ms" << std::endl;
      int size = last_query_amount * sizeof(VoxelSpatialTSDF);
      mReconstrMsg->data.resize(size);
      std::memcpy(&(mReconstrMsg->data[0]),
                  (char*)mSemanticReconstr.data(),
                  last_query_amount * sizeof(VoxelSpatialTSDF));

      tsdfCb(mReconstrMsg);
      ros::spinOnce();
      rate.sleep();
    }
  });

  std::thread t_pose([&](){
    static ros::Time stamp;
    static ros::Rate rate(30);
    while(ros::ok())
    {
      u_int64_t t_query = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
      SE3<float> mSlamPose = my_sys->query_camera_pose(t_query);

      Eigen::Quaternion<float> R = mSlamPose.GetR();
      Eigen::Matrix<float, 3, 1> T = mSlamPose.GetT();
      // std::cout<<"Queried pose at "<<t_query<<std::endl;
      // std::cout<<"Rotation: "<<R.x()<<", "<< R.y()<<", "<< R.z()<<", "<<R.w()<<", "<<std::endl;
      // std::cout<<"Translation: "<<T.x()<<", "<< T.y()<<", "<< T.z()<<", "<<std::endl;

      tf2::Transform tf2_trans;
      tf2::Transform tf2_trans_inv;
      tf2_trans.setRotation(tf2::Quaternion(R.x(), R.y(), R.z(), R.w()));
      tf2_trans.setOrigin(tf2::Vector3(T.x(), T.y(), T.z()));

      stamp.sec  = t_query / 1000; //s
      stamp.nsec = (t_query % 1000) * 1000 * 1000; //ns
      transformStamped.header.stamp = stamp;
      transformStamped.header.frame_id = "slam";
      transformStamped.child_frame_id = "zed";
      tf2_trans_inv = tf2_trans.inverse();
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
  zed_mask_lock.lock();
  zedLeftMask = cv_bridge::toCvShare(msg, "8UC1")->image.clone();
  zed_mask_lock.unlock();
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
  mask_lock.lock();
  l515Mask = cv_bridge::toCvShare(msg,  "8UC1")->image.clone();
  mask_lock.unlock();

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
  image_transport::Subscriber sub1 = it.subscribe("/robot_mask/zed_slam_left", 1, zedMaskCb);
  image_transport::Subscriber sub2 = it.subscribe("/robot_mask/realsense_slam_l515", 1, l515MaskCb);

  tf2_ros::TransformBroadcaster mTfSlam;

  visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("world","/mesh_visual", mNh));
  visual_tools_->setPsychedelicMode(false);
  visual_tools_->loadMarkerPub();
  tfListener= std::make_shared<tf2_ros::TransformListener>(tfBuffer);
  meshPub = mNh.advertise<shape_msgs::Mesh>("/mesh", 1);
  mReconstrMsg.reset(new std_msgs::Float32MultiArray);

  std::string model_path, calib_path, orb_vocab_path;
  int devid;
  bool renderFlag;
  bool global_mesh;
  bool require_mesh;
  double bbox_xy;

  mNh.getParam("/ros_disinf_slam/model_path", model_path); 
  mNh.getParam("/ros_disinf_slam/calib_path", calib_path); 
  mNh.getParam("/ros_disinf_slam/orb_vocab_path", orb_vocab_path); 
  mNh.param("/ros_disinf_slam/devid", devid, 2);
  mNh.param("/ros_disinf_slam/bbox_xy", bbox_xy, 4.0);
  mNh.param("/ros_disinf_slam/renderer", renderFlag, false);
  mNh.param("/ros_disinf_slam/global_mesh", global_mesh, true);
  mNh.param("/ros_disinf_slam/require_mesh", require_mesh, true);


  // ROS_INFO_STREAM(calib_path);

  auto cfg = GetAndSetConfig(calib_path);
  // initialize cameras
  ZEDNative zed_native(*cfg, devid);
  L515 l515;
  // initialize slam
  std::shared_ptr<DISINFSystem> my_system = std::make_shared<DISINFSystem>(
      calib_path, orb_vocab_path, model_path, renderFlag
  );

  run(zed_native, l515, my_system, mNh, mTfSlam, renderFlag, bbox_xy, global_mesh);

  return EXIT_SUCCESS;
}
