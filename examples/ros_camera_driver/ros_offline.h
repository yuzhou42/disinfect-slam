#include <iostream>
#include <string>
#include <thread>

// #include <openvslam/system.h>
// #include <popl.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "disinfect_slam/disinfect_slam.h"
#include "utils/config_reader.hpp"
#include "utils/time.hpp"
#include "cameras/zed.h"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

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
#include "System.h"
#include <opencv2/core/eigen.hpp>
#include "utils/rotation_math/pose_manager.h"


using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

class ImuGrabber {
 public:
  ImuGrabber() {
    imuBuf = queue<sensor_msgs::ImuConstPtr>();
    std::cout << "initialize imu grabber" << std::endl;
  };

  void GrabImu(const sensor_msgs::ImuConstPtr& imu_msg);

  queue<sensor_msgs::ImuConstPtr> imuBuf;
  std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(){
        imgLeftBuf = queue<sensor_msgs::ImageConstPtr>();
        imgRightBuf = queue<sensor_msgs::ImageConstPtr>();
        imgDepthBuf = queue<sensor_msgs::ImageConstPtr>();
        imgRgbBuf = queue<sensor_msgs::ImageConstPtr>();
        std::cout<<"initialize image grabber"<<std::endl;
    };
    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageDepth(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRgb(const sensor_msgs::ImageConstPtr& msg);
    // double syncImages(  queue<sensor_msgs::ImageConstPtr>& imagebuf1,  queue<sensor_msgs::ImageConstPtr>& imagebuf2, std::mutex& lock1, std::mutex& lock2, cv::Mat& image1, cv::Mat& image2);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg, string type = "rgb8");
    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf, imgDepthBuf, imgRgbBuf;
    std::mutex mBufMutexLeft,mBufMutexRight,mBufMutexDepth, mBufMutexRgb;
   
    // const bool mbClahe;
    // cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};

class SyncSubscriber
{
public:
    SyncSubscriber();
    // void stereoCb(const ImageConstPtr& stereoLeft, const ImageConstPtr& stereoRight);
    // void ImuCb(const sensor_msgs::ImuConstPtr &imu_msg);
    // void stereoCb(const ImageConstPtr& stereoLeft, const ImageConstPtr& stereoRight, const ImageConstPtr& maskLeft);

    void depthCb(const ImageConstPtr& rgbImg, const ImageConstPtr& depth);
    void depthCb(const ImageConstPtr& rgbImg, const ImageConstPtr& depth, const ImageConstPtr& maskDepth);

    void reconstTimerCallback(const ros::TimerEvent&);
    void poseTimerCallback(const ros::TimerEvent&);
    void tsdfCb(std::vector<VoxelSpatialTSDF> & SemanticReconstr);
    void slamTh();
    void reconstTh();

private:
    ros::NodeHandle nh_;
    std::string model_path, calib_path, orb_vocab_path;
    bool use_mask;
    bool global_mesh;
    bool pangolin_view;
    bool do_rectify;
    int sensor, debug_out;
    float truncation_distance;
    float cell_size;
    std::vector<float> query_bbox;
    ros::Publisher mPubTsdfGlobal;
    ros::Publisher mPubTsdfLocal;
    geometry_msgs::TransformStamped transformStamped;
    std_msgs::Float32MultiArray::Ptr mReconstrMsg;
    geometry_msgs::Pose T_ws;
    // initialize slam
    std::shared_ptr<DISINFSystem> my_sys;
    // std::shared_ptr<ORB_SLAM3::System> mpSLAM; 
    // message_filters::Subscriber<sensor_msgs::Image> sub_1_;
    // message_filters::Subscriber<sensor_msgs::Image> sub_2_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>
        MySyncPolicy2;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image>
        MySyncPolicy3;
    typedef message_filters::Synchronizer<MySyncPolicy2> Sync2;
    typedef message_filters::Synchronizer<MySyncPolicy3> Sync3;

    boost::shared_ptr<Sync2> sync2_stereo_;
    boost::shared_ptr<Sync2> sync2_depth_;

    boost::shared_ptr<Sync3> sync3_stereo_;
    boost::shared_ptr<Sync3> sync3_depth_;

    image_transport::SubscriberFilter stereoLeft, stereoRight, depth, rgbImg, maskLeft, maskDepth;
    
    int64_t tLastUpdate; // Last updated ros time, ns
    ros::Timer reconstTimer;
    ros::Timer poseTimer;

    rviz_visual_tools::RvizVisualToolsPtr visual_tools_;
    tf2_ros::Buffer tfBuffer;
    ros::Publisher meshPub;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    cv::Mat M1l,M2l,M1r,M2r;
    cv::Mat img_left, img_right;
    cv::Mat img_rgb;
    cv::Mat img_depth;
    queue<sensor_msgs::ImuConstPtr> imuBuf;    //IMU temp buffer
    std::mutex mBufMutex; //lock for imu buffer
    std::mutex mDepthMutex; //lock for depth  

    std::shared_ptr<ImuGrabber> mpImuGb;
    std::shared_ptr<ImageGrabber> mpIgb;
    // ImuGrabber* mpImuGb;
    // ImageGrabber* mpIgb;
    std::thread mslamTh;
    std::thread mReconstTh;

    ros::Subscriber sub_imu;
    ros::Subscriber sub_img_left;
    ros::Subscriber sub_img_right;
    ros::Subscriber sub_img_depth;
    ros::Subscriber sub_img_rgb;
    cv::Mat Tcw;
    ORB_SLAM3::System::eSensor mSensor;
    vector<ORB_SLAM3::IMU::Point> vImuMeas;

};

