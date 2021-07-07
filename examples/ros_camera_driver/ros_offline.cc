#include "ros_offline.h"

SyncSubscriber::SyncSubscriber()
{
    nh_.getParam("/ros_offline_orb3/model_path", model_path);
    nh_.getParam("/ros_offline_orb3/calib_path", calib_path);
    nh_.getParam("/ros_offline_orb3/orb_vocab_path", orb_vocab_path);
    nh_.param("/ros_offline_orb3/bbox_xy", bbox_xy, 4.0);
    nh_.param("/ros_offline_orb3/renderer", renderFlag, false);
    nh_.param("/ros_offline_orb3/use_mask", use_mask, false);
    nh_.param("/ros_offline_orb3/global_mesh", global_mesh, true);
    nh_.param("/ros_offline_orb3/do_rectify", do_rectify, true);

    image_transport::ImageTransport it(nh_);

    // stereoLeft.subscribe(it, "/stereoLeft", 3);
    // stereoRight.subscribe(it, "/stereoRight", 3);
    depth.subscribe(it, "/depth", 3);
    rgbImg.subscribe(it, "/rgbImg", 3);
    // maskLeft.subscribe(it, "/maskLeft", 3);
    maskDepth.subscribe(it, "/maskDepth", 3);

    mpImuGb = std::make_shared<ImuGrabber>();
    mpIgb = std::make_shared<ImageGrabber>();
    // mpImuGb = new ImuGrabber();
    // mpIgb = new ImageGrabber();
    sub_imu = nh_.subscribe("/zed2/zed_node/imu/data", 10, &ImuGrabber::GrabImu, this->mpImuGb.get()); 
    sub_img_left = nh_.subscribe("/zed2/zed_node/left_raw/image_raw_color", 5, &ImageGrabber::GrabImageLeft, this->mpIgb.get() );
    sub_img_right = nh_.subscribe("/zed2/zed_node/right_raw/image_raw_color", 5, &ImageGrabber::GrabImageRight, this->mpIgb.get());

    // publishers
    // mPubTsdfGlobal = nh_.advertise<std_msgs::Float32MultiArray>("/tsdf_global", 4);
    // mPubTsdfLocal  = nh_.advertise<std_msgs::Float32MultiArray>("/tsdf_local", 4);
    // if (use_mask)
    // {
    //     sync3_depth_.reset(new Sync3(MySyncPolicy3(10), rgbImg, depth, maskDepth));
    //     sync3_depth_->registerCallback(boost::bind(&SyncSubscriber::depthCb, this, _1, _2, _3));
    // }
    // else
    // {
    //     sync2_depth_.reset(new Sync2(MySyncPolicy2(10), rgbImg, depth));
    //     sync2_depth_->registerCallback(boost::bind(&SyncSubscriber::depthCb, this, _1, _2));
    // }

    // sync2_stereo_.reset(new Sync2(MySyncPolicy2(10), stereoLeft, stereoRight));
    // sync2_stereo_->registerCallback(boost::bind(&SyncSubscriber::stereoCb, this, _1, _2));

    // reconstTimer = nh_.createTimer(ros::Duration(0.2), &SyncSubscriber::reconstTimerCallback, this);
    // poseTimer    = nh_.createTimer(ros::Duration(0.05), &SyncSubscriber::poseTimerCallback, this);

    // ROS_INFO_STREAM(calib_path);

    // auto cfg = GetAndSetConfig(calib_path);
    doRectify(calib_path, &M1l, &M1r, &M2l, &M2r);
    my_sys   = std::make_shared<DISINFSystem>(calib_path, orb_vocab_path, model_path, renderFlag);
    mpSLAM = std::make_shared<ORB_SLAM3::System>(orb_vocab_path, calib_path,
                         ORB_SLAM3::System::IMU_STEREO, renderFlag);

    mReconstrMsg.reset(new std_msgs::Float32MultiArray);
    visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("world","/mesh_visual", nh_));
    visual_tools_->setPsychedelicMode(false);
    visual_tools_->loadMarkerPub();
    tfListener= std::make_shared<tf2_ros::TransformListener>(tfBuffer);
    meshPub = nh_.advertise<shape_msgs::Mesh>("/mesh", 1);

    try{
        geometry_msgs::TransformStamped transformStampedInit = tfBuffer.lookupTransform("world", "slam", ros::Time::now(), ros::Duration(5));
        // T_ws = tf2::transformToEigen(transformStampedInit);
        T_ws.position.x = transformStampedInit.transform.translation.x;
        T_ws.position.y = transformStampedInit.transform.translation.y;
        T_ws.position.z = transformStampedInit.transform.translation.z;

        T_ws.orientation =  transformStampedInit.transform.rotation;
        std::cout<<"Init world slam transform!"<<std::endl;
        }
        catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());   
        }
    mslamTh = std::thread(&SyncSubscriber::slamTh, this);

}

void SyncSubscriber::slamTh()
{
  const double maxTimeDiff = 0.01;
//   std::thread t_slam([&](){
  while(1)
  {
    cv::Mat imLeft, imRight;
    double tImLeft = 0, tImRight = 0;
    if (!mpIgb->imgLeftBuf.empty()&&!mpIgb->imgRightBuf.empty()&&!mpImuGb->imuBuf.empty())
    {
      tImLeft = mpIgb->imgLeftBuf.front()->header.stamp.toSec();
      tImRight = mpIgb->imgRightBuf.front()->header.stamp.toSec();

      mpIgb->mBufMutexRight.lock();
      while((tImLeft-tImRight)>maxTimeDiff && mpIgb->imgRightBuf.size()>1)
      {
        mpIgb->imgRightBuf.pop();
        tImRight = mpIgb->imgRightBuf.front()->header.stamp.toSec();
      }
      mpIgb->mBufMutexRight.unlock();

      mpIgb->mBufMutexLeft.lock();
      while((tImRight-tImLeft)>maxTimeDiff && mpIgb->imgLeftBuf.size()>1)
      {
        mpIgb->imgLeftBuf.pop();
        tImLeft = mpIgb->imgLeftBuf.front()->header.stamp.toSec();
      }
      mpIgb->mBufMutexLeft.unlock();

      if((tImLeft-tImRight)>maxTimeDiff || (tImRight-tImLeft)>maxTimeDiff)
      {
        // std::cout << "big time difference" << std::endl;
        continue;
      }
      if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec())
          continue;

      mpIgb->mBufMutexLeft.lock();
      imLeft = mpIgb->GetImage(mpIgb->imgLeftBuf.front());
      mpIgb->imgLeftBuf.pop();
      mpIgb->mBufMutexLeft.unlock();

      mpIgb->mBufMutexRight.lock();
      imRight = mpIgb->GetImage(mpIgb->imgRightBuf.front());
      mpIgb->imgRightBuf.pop();
      mpIgb->mBufMutexRight.unlock();

      vector<ORB_SLAM3::IMU::Point> vImuMeas;
      mpImuGb->mBufMutex.lock();
      if(!mpImuGb->imuBuf.empty())
      {
        // Load imu measurements from buffer
        vImuMeas.clear();
        while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImLeft)
        {
          double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
          // std::cout<<"t_imu: "<<t<<std::endl;
          cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
          cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
          mpImuGb->imuBuf.pop();
        }
      }
      mpImuGb->mBufMutex.unlock();

      if(do_rectify)
      {
        cv::remap(imLeft,imLeft,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRight,M1r,M2r,cv::INTER_LINEAR);
      }

//     // my_sys->feed_stereo_IMU(img_left, img_right, int64_t(timeDiff * 1e9), vImuMeas);
      Tcw = mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
      // std::cout << "Tcw: " << Tcw << std::endl;
      // if(!Tcw.empty()){
      //   Eigen::Matrix<float, 4, 4> eigenT;
      //   cv::cv2eigen(Tcw, eigenT);
      //   // std::cout<<"eigenT: "<<eigenT<<std::endl;
      //   const SE3<float> posecam_P_world(eigenT);
      //   my_sys->camera_pose_manager_->register_valid_pose(tImLeft*1e3, posecam_P_world); //s to ms
      // }
      
    //   geometry_msgs::PoseStamped pose;
    //   pose.header.stamp = ros::Time::now();
    //   pose.header.frame_id ="map";

    //   if(!Tcw.empty()){
    //     cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t(); // Rotation information
    //     cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3); // translation information
    //     vector<float> q = ORB_SLAM3::Converter::toQuaternion(Rwc);

    //     tf::Transform new_transform;
    //     new_transform.setOrigin(tf::Vector3(twc.at<float>(0, 0), twc.at<float>(0, 1), twc.at<float>(0, 2)));

    //     tf::Quaternion quaternion(q[0], q[1], q[2], q[3]);
    //     new_transform.setRotation(quaternion);

    //     tf::poseTFToMsg(new_transform, pose.pose);
    //     pose_pub.publish(pose);


    //   }

      std::chrono::milliseconds tSleep(1);
      std::this_thread::sleep_for(tSleep);
    }
  }
}

void SyncSubscriber::depthCb(const ImageConstPtr& rgbImg, const ImageConstPtr& depth)
{
    cv::Mat img_rgb         = cv_bridge::toCvShare(rgbImg, "rgb8")->image.clone();
    cv::Mat img_depth       = cv_bridge::toCvShare(depth, "16UC1")->image.clone(); //mono16/16UC1
    // static double initTime = rgbImg->header.stamp.toSec()*1e9;
    double timestamp = rgbImg->header.stamp.toSec()*1e3;
    // double timeDiff = timestamp-initTime;
    my_sys->feed_rgbd_frame(img_rgb, img_depth, int64_t(timestamp));
}

void SyncSubscriber::depthCb(const ImageConstPtr& rgbImg,
                            const ImageConstPtr& depth,
                            const ImageConstPtr& maskDepth)
{
    ROS_INFO("got stereo data");
    cv::Mat img_rgb   = cv_bridge::toCvShare(rgbImg, "rgb8")->image.clone();
    cv::Mat img_depth = cv_bridge::toCvShare(depth, "16UC1")->image.clone();
    cv::Mat l515MaskL = cv_bridge::toCvShare(maskDepth, "8UC1")->image.clone();
    static double initTime = rgbImg->header.stamp.toSec()*1e9;
    const int64_t timestamp = rgbImg->header.stamp.toSec()*1e9;
    double timeDiff = timestamp-initTime;

    my_sys->feed_rgbd_frame(img_rgb, img_depth, int64_t(timeDiff), l515MaskL);
}

void SyncSubscriber::tsdfCb(std::vector<VoxelSpatialTSDF> & SemanticReconstr)
{
    const auto st = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
    int numPoints = SemanticReconstr.size();
    float minValue = 1e100, maxValue = -1e100;
    Math3D::AABB3D bbox;
    for(int i=0;i<numPoints;i++) {
        minValue = Min(minValue,SemanticReconstr[i].tsdf);
        maxValue = Max(maxValue,SemanticReconstr[i].tsdf);
        bbox.expand(Math3D::Vector3(SemanticReconstr[i].position[0],SemanticReconstr[i].position[1],SemanticReconstr[i].position[2]));
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
    Math3D::Vector3 ofs(CELL_SIZE*0.5);
    for(int i=0;i<numPoints;i++) {
        tsdf.tsdf.SetValue(Math3D::Vector3(SemanticReconstr[i].position[0],SemanticReconstr[i].position[1],SemanticReconstr[i].position[2])+ofs,SemanticReconstr[i].tsdf);
    }

    Meshing::TriMesh mesh;
    tsdf.ExtractMesh(mesh);
    // std::cout<<"Before Merge: trisSize: "<<mesh.tris.size()<<std::endl;
    // MergeVertices(mesh, 0.05);
    int vertsSize = mesh.verts.size();
    int trisSize = mesh.tris.size();
    std::cout<<"trisSize: "<<trisSize<<std::endl;
    const auto end = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
    std::cout<<"mesh processing time: "<<end-st<<" ms"<<std::endl;
    shape_msgs::Mesh::Ptr mMeshMsg = boost::make_shared<shape_msgs::Mesh>();
    // geometry_msgs/Point[] 
    mMeshMsg->vertices.resize(vertsSize);
    // shape_msgs/MeshTriangle[] 
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
    visual_tools_->publishMesh(T_ws, *mMeshMsg, rviz_visual_tools::ORANGE, 1, "mesh", 1); // rviz_visual_tools::TRANSLUCENT_LIGHT
    // Don't forget to trigger the publisher!
    visual_tools_->trigger();
}

void SyncSubscriber::reconstTimerCallback(const ros::TimerEvent&)
{
    static unsigned int last_query_time = 0;
    static size_t last_query_amount     = 0;
    // static float bbox = 4.0;
    static float x_range[2]           = {-bbox_xy, bbox_xy};
    static float y_range[2]           = {-4, 2};
    static float z_range[2]           = {-bbox_xy, bbox_xy};
    static BoundingCube<float> volumn = {
        x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]};

    if (!global_mesh)
    {
        const auto st = GetTimestamp<std::chrono::milliseconds>(); // nsec
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
    tsdfCb(mSemanticReconstr);

}

void SyncSubscriber::poseTimerCallback(const ros::TimerEvent&)
{
    static tf2_ros::TransformBroadcaster mTfSlam;
    static ros::Time stamp;
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
}

void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg)
{
  // std::cout<<"got image"<<std::endl;
  mBufMutexLeft.lock();
//   if (!imgLeftBuf.empty())
//     imgLeftBuf.pop();
  imgLeftBuf.push(img_msg);
  mBufMutexLeft.unlock();
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg)
{
  // std::cout<<"got image"<<std::endl;
  mBufMutexRight.lock();
//   if (!imgRightBuf.empty())
//     imgRightBuf.pop();
  imgRightBuf.push(img_msg);
  mBufMutexRight.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  
  if(cv_ptr->image.type()==0)
  {
    return cv_ptr->image.clone();
  }
  else
  {
    std::cout << "Error type" << std::endl;
    return cv_ptr->image.clone();
  }
}


void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{

  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}


// void SyncSubscriber::stereoCb(const ImageConstPtr& stereoLeft,
//                              const ImageConstPtr& stereoRight,
//                              const ImageConstPtr& maskLeft)
// {
//     ROS_INFO("got stereo data");
//     cv::Mat img_left     = cv_bridge::toCvShare(stereoLeft, "bgr8")->image;
//     cv::Mat img_right    = cv_bridge::toCvShare(stereoRight, "bgr8")->image;
//     cv::Mat zedLeftMaskL = cv_bridge::toCvShare(maskLeft, "8UC1")->image;
//     static double initTime = stereoLeft->header.stamp.toSec()*1000;
//     const int64_t timestamp = stereoLeft->header.stamp.toSec();
//     double timeDiff = timestamp-initTime;

//     my_sys->feed_stereo_frame(img_left, img_right, int64_t(timeDiff), zedLeftMaskL);
// }

// void SyncSubscriber::stereoCb(const ImageConstPtr& stereoLeft, const ImageConstPtr& stereoRight)
// {
//     img_left        = cv_bridge::toCvShare(stereoLeft, sensor_msgs::image_encodings::MONO8)->image.clone();
//     img_right       = cv_bridge::toCvShare(stereoRight, sensor_msgs::image_encodings::MONO8)->image.clone();
//     cv::remap(img_left,img_left,M1l,M2l,cv::INTER_LINEAR);
//     cv::remap(img_right,img_right,M1r,M2r,cv::INTER_LINEAR);
//     static double initTime = stereoLeft->header.stamp.toSec();
//     double timestamp = stereoLeft->header.stamp.toSec();
//     double timeDiff = timestamp-initTime;
//     std::cout<<"Image time: "<<timeDiff<<std::endl;
//     // cv::imshow("zed_left", img_left);
//     // cv::waitKey(0);
//     // cv::imshow("zed_right", img_right);
//     // cv::waitKey(0);
//     // ROS_INFO("Stereo timestamp: %f, %d", timeDiff, int64_t(timeDiff));
//     // my_sys->feed_stereo_frame(img_left, img_right, int64_t(timeDiff));

//     vector<ORB_SLAM3::IMU::Point> vImuMeas;
//     mBufMutex.lock();
//     if(!imuBuf.empty())
//     {
//         // Load imu measurements from buffer
//         vImuMeas.clear();
//         while(!imuBuf.empty() && imuBuf.front()->header.stamp.toSec()<=stereoLeft->header.stamp.toSec())
//         {
//             double t = imuBuf.front()->header.stamp.toSec() - initTime;
//             if(t<0) 
//             {
//                 imuBuf.pop();
//                 continue;
//             }
//             std::cout<<"Imu time: "<<t<<std::endl;
//             cv::Point3f acc(imuBuf.front()->linear_acceleration.x, imuBuf.front()->linear_acceleration.y, imuBuf.front()->linear_acceleration.z);
//             cv::Point3f gyr(imuBuf.front()->angular_velocity.x, imuBuf.front()->angular_velocity.y, imuBuf.front()->angular_velocity.z);
//             vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
//             imuBuf.pop();
//         }
//     }
//     mBufMutex.unlock();
//     // my_sys->feed_stereo_IMU(img_left, img_right, int64_t(timeDiff * 1e9), vImuMeas);
//     cv::Mat Tcw = SLAM->TrackStereo(img_left,img_right,timeDiff,vImuMeas);
//     // cv::Mat Tcw = my_sys->SLAM_->TrackStereo(img_left,img_right,timeDiff);
// }

// void SyncSubscriber::ImuCb(const sensor_msgs::ImuConstPtr &imu_msg)
//     {
//         // mBufMutex.lock();
//         // std::cout<<"get imu data"<<std::endl;
//         imuBuf.push(imu_msg);
//         // mBufMutex.unlock();
//         // return;
//     }