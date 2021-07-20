#include "ros_offline.h"

SyncSubscriber::SyncSubscriber()
{
    nh_.getParam("/debug_level", debug_out);  // 1: debug, 2: info
    spdlog::set_level(spdlog::level::level_enum(debug_out)); 
    nh_.getParam("/ros_offline_orb3/model_path", model_path);
    nh_.getParam("/ros_offline_orb3/calib_path", calib_path);
    nh_.getParam("/ros_offline_orb3/orb_vocab_path", orb_vocab_path);
    nh_.getParam("/pangolin_view", pangolin_view);
    nh_.getParam("/use_mask", use_mask);
    nh_.getParam("/global_mesh", global_mesh);
    nh_.getParam("/do_rectify", do_rectify);
    nh_.getParam("/truncation_distance", truncation_distance);
    nh_.getParam("/cell_size", cell_size);
    nh_.getParam("/sensor", sensor); // 0: mono, 1: stereo, 2: rgbd, 3: imu_mono, 4: imu_stereo
    nh_.getParam("/query_bbox", query_bbox);
    spdlog::info("query bbox size: x: [{},{}],  y: [{},{}],  z: [{},{}]", 
    query_bbox[0], query_bbox[1], query_bbox[2], query_bbox[3],query_bbox[4],query_bbox[5]);
    switch (sensor)
    {
    case 1:
      mSensor = ORB_SLAM3::System::STEREO;
      spdlog::info("Stereo");
      break;
    case 4:
      mSensor = ORB_SLAM3::System::IMU_STEREO;
      spdlog::info("Stereo - inertial");
      break;
    }
    // image_transport::ImageTransport it(nh_);
    // stereoLeft.subscribe(it, "/stereoLeft", 3);
    // stereoRight.subscribe(it, "/stereoRight", 3);
    // depth.subscribe(it, "/depth", 3);
    // rgbImg.subscribe(it, "/rgbImg", 3);
    // maskLeft.subscribe(it, "/maskLeft", 3);
    // maskDepth.subscribe(it, "/maskDepth", 3);

    mpImuGb = std::make_shared<ImuGrabber>();
    mpIgb = std::make_shared<ImageGrabber>();
    // mpImuGb = new ImuGrabber();
    // mpIgb = new ImageGrabber();
    sub_imu = nh_.subscribe("/zed2/zed_node/imu/data", 10, &ImuGrabber::GrabImu, this->mpImuGb.get()); 
    sub_img_left = nh_.subscribe("/zed2/zed_node/left_raw/image_raw_color", 5, &ImageGrabber::GrabImageLeft, this->mpIgb.get() );
    sub_img_right = nh_.subscribe("/zed2/zed_node/right_raw/image_raw_color", 5, &ImageGrabber::GrabImageRight, this->mpIgb.get());
    sub_img_depth = nh_.subscribe("/camera/aligned_depth_to_color/image_raw", 5, &ImageGrabber::GrabImageDepth, this->mpIgb.get() );
    sub_img_rgb = nh_.subscribe("/camera/color/image_raw", 5, &ImageGrabber::GrabImageRgb, this->mpIgb.get());
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

    reconstTimer = nh_.createTimer(ros::Duration(0.2), &SyncSubscriber::reconstTimerCallback, this);
    poseTimer    = nh_.createTimer(ros::Duration(0.05), &SyncSubscriber::poseTimerCallback, this);

    doRectify(calib_path, &M1l, &M1r, &M2l, &M2r);
    my_sys   = std::make_shared<DISINFSystem>(calib_path, orb_vocab_path, model_path, mSensor, pangolin_view);

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
        spdlog::info("Init world slam transform!");
        }
        catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());   
        }
    mslamTh = std::thread(&SyncSubscriber::slamTh, this);
    mReconstTh = std::thread(&SyncSubscriber::reconstTh, this);

}

void SyncSubscriber::reconstTh()
{
  spdlog::info("Start reconstruction thread");
  const double maxTimeDiff = 0.05;
  
  while(true){
    cv::Mat imDepth, imRgb;
    double tlmDepth = 0, tlmRgb = 0;
    if (!mpIgb->imgDepthBuf.empty()&&!mpIgb->imgRgbBuf.empty())
    {
      tlmDepth = mpIgb->imgDepthBuf.front()->header.stamp.toSec();
      tlmRgb = mpIgb->imgRgbBuf.front()->header.stamp.toSec();

      mpIgb->mBufMutexRgb.lock();
      while((tlmDepth-tlmRgb)>maxTimeDiff && mpIgb->imgRgbBuf.size()>1)
      {
        mpIgb->imgRgbBuf.pop();
        tlmRgb = mpIgb->imgRgbBuf.front()->header.stamp.toSec();
      }
      mpIgb->mBufMutexRgb.unlock();

      mpIgb->mBufMutexDepth.lock();
      while((tlmRgb-tlmDepth)>maxTimeDiff && mpIgb->imgDepthBuf.size()>1)
      {
        mpIgb->imgDepthBuf.pop();
        tlmDepth = mpIgb->imgDepthBuf.front()->header.stamp.toSec();
      }
      mpIgb->mBufMutexDepth.unlock();

      if((tlmDepth-tlmRgb)>maxTimeDiff || (tlmRgb-tlmDepth)>maxTimeDiff)
      {
        spdlog::debug("big time difference");
        continue;
      }

      mpIgb->mBufMutexDepth.lock();
      imDepth = mpIgb->GetImage(mpIgb->imgDepthBuf.front(), "16UC1");
      mpIgb->imgDepthBuf.pop();
      mpIgb->mBufMutexDepth.unlock();

      mpIgb->mBufMutexRgb.lock();
      imRgb = mpIgb->GetImage(mpIgb->imgRgbBuf.front(),"rgb8");
      mpIgb->imgRgbBuf.pop();
      mpIgb->mBufMutexRgb.unlock();

      my_sys->feed_rgbd_frame(imRgb, imDepth, int64_t(tlmDepth*1e3));
    }
  }
}

void SyncSubscriber::slamTh()
{
  spdlog::info("Start SLAM thread");
  const double maxTimeDiff = 0.01;

  while(1)
  {
    cv::Mat imLeft, imRight;
    double tImLeft = 0, tImRight = 0;
    if (!mpIgb->imgLeftBuf.empty()&&!mpIgb->imgRightBuf.empty()&&(!mpImuGb->imuBuf.empty() || mSensor != ORB_SLAM3::System::IMU_STEREO ))
    {
      tImLeft = mpIgb->imgLeftBuf.front()->header.stamp.toSec();
      tImRight = mpIgb->imgRightBuf.front()->header.stamp.toSec();

      tLastUpdate = (int64_t) tImLeft * 1e6;

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
        spdlog::info("big time difference for stereo");
        continue;
      }

      mpIgb->mBufMutexLeft.lock();
      imLeft = mpIgb->GetImage(mpIgb->imgLeftBuf.front());
      mpIgb->imgLeftBuf.pop();
      mpIgb->mBufMutexLeft.unlock();

      mpIgb->mBufMutexRight.lock();
      imRight = mpIgb->GetImage(mpIgb->imgRightBuf.front());
      mpIgb->imgRightBuf.pop();
      mpIgb->mBufMutexRight.unlock();

      if(do_rectify)
      {
        cv::remap(imLeft,imLeft,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRight,M1r,M2r,cv::INTER_LINEAR);
      }

      if(mSensor == ORB_SLAM3::System::IMU_STEREO)
      {
        if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec())
          continue;
        mpImuGb->mBufMutex.lock();
        if(!mpImuGb->imuBuf.empty())
        {
          // Load imu measurements from buffer
          vImuMeas.clear();
          while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImLeft)
          {
            double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
            cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
            cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
            vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
            mpImuGb->imuBuf.pop();
          }
        }
        mpImuGb->mBufMutex.unlock();
        // std::cout<<vImuMeas.size()<<std::endl;
        my_sys->feed_stereo_IMU(imLeft, imRight, tImLeft, vImuMeas);
      }
      else if(mSensor == ORB_SLAM3::System::STEREO)
        my_sys->feed_stereo(imLeft, imRight, tImLeft);
        // my_sys->feed_stereo(imLeft, imRight, tImLeft, vImuMeas);

      // Tcw = mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
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
    if(truncation_distance < 0) {
        //auto-detect truncation distance
        truncation_distance = Max(-minValue,maxValue)*0.99;
        spdlog::debug("Auto-detected truncation distance {}", truncation_distance);
    }
    // printf("Using cell size %g\n",cell_size);
    Geometry::SparseTSDFReconstruction tsdf(Math3D::Vector3(cell_size),truncation_distance);
    tsdf.tsdf.defaultValue[0] = truncation_distance;
    Math3D::Vector3 ofs(cell_size*0.5);
    for(int i=0;i<numPoints;i++) {
        tsdf.tsdf.SetValue(Math3D::Vector3(SemanticReconstr[i].position[0],SemanticReconstr[i].position[1],SemanticReconstr[i].position[2])+ofs,SemanticReconstr[i].tsdf);
    }

    Meshing::TriMesh mesh;
    tsdf.ExtractMesh(mesh);
    // std::cout<<"Before Merge: trisSize: "<<mesh.tris.size()<<std::endl;
    // MergeVertices(mesh, 0.05);
    int vertsSize = mesh.verts.size();
    int trisSize = mesh.tris.size();
    spdlog::debug("triSize: {}", trisSize);
    const auto end = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
    spdlog::debug("mesh processing time: {} ms", end-st);
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
    static BoundingCube<float> volumn = {
        query_bbox[0], query_bbox[1], query_bbox[2], query_bbox[3],query_bbox[4],query_bbox[5]};

    if (!global_mesh)
    {
        // const auto st = GetTimestamp<std::chrono::milliseconds>(); // nsec
        float x_off   = transformStamped.transform.translation.x,
              y_off   = transformStamped.transform.translation.y,
              z_off   = transformStamped.transform.translation.z;
        spdlog::info("x_off: {}, y_off: {}, z_off: {}", x_off, y_off, z_off);
        volumn = {x_off + query_bbox[0],
                    x_off + query_bbox[1],
                    y_off + query_bbox[2],
                    y_off + query_bbox[3],
                    z_off + query_bbox[4],
                    z_off + query_bbox[5]};
    }

    const auto st          = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
    auto mSemanticReconstr           = my_sys->query_tsdf(volumn);
    const auto end                   = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
    last_query_time                  = end - st;
    last_query_amount                = mSemanticReconstr.size();
    spdlog::debug("Last queried {} voxels, took {} ms", last_query_amount, last_query_time);
    tsdfCb(mSemanticReconstr);

}

void SyncSubscriber::poseTimerCallback(const ros::TimerEvent&) {
  static tf2_ros::TransformBroadcaster mTfSlam;
  static ros::Time stamp;

  u_int64_t t_query = tLastUpdate;
  SE3<float> mSlamPose = my_sys->query_camera_pose(t_query * 1e3);

  Eigen::Quaternion<float> R = mSlamPose.GetR();
  Eigen::Matrix<float, 3, 1> T = mSlamPose.GetT();
  // std::cout<<"Queried pose at "<<t_query<<std::endl;
  // std::cout<<"Rotation: "<<R.x()<<", "<< R.y()<<", "<< R.z()<<", "<<R.w()<<", "<<std::endl;
  // std::cout<<"Translation: "<<T.x()<<", "<< T.y()<<", "<< T.z()<<", "<<std::endl;

  tf2::Transform tf2_trans;
  tf2::Transform tf2_trans_inv;
  tf2_trans.setRotation(tf2::Quaternion(R.x(), R.y(), R.z(), R.w()));
  tf2_trans.setOrigin(tf2::Vector3(T.x(), T.y(), T.z()));

  u_int64_t t_publish = (int64_t) GetSystemTimestamp<std::chrono::nanoseconds>();
  stamp.sec = t_publish / 1e6;
  stamp.nsec = t_publish;

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

void ImageGrabber::GrabImageDepth(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexDepth.lock();
//   if (!imgLeftBuf.empty())
//     imgLeftBuf.pop();
  imgDepthBuf.push(img_msg);
  mBufMutexDepth.unlock();
}

void ImageGrabber::GrabImageRgb(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexRgb.lock();
//   if (!imgRightBuf.empty())
//     imgRightBuf.pop();
  imgRgbBuf.push(img_msg);
  mBufMutexRgb.unlock();
}


cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg, string type)
{
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, type);
    // sensor_msgs::image_encodings::MONO8
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  return cv_ptr->image.clone();
}


void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
  // spdlog::debug("got imu");

  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}

// double ImageGrabber::syncImages(queue<sensor_msgs::ImageConstPtr>& imagebuf1, queue<sensor_msgs::ImageConstPtr>& imagebuf2, std::mutex& lock1, std::mutex& lock2,  cv::Mat& image1, cv::Mat& image2 )
// {
//   const double maxTimeDiff = 0.01;
//   double t1 = 0, t2 = 0;
//   if (!imagebuf1.empty()&&!imagebuf2.empty())
//     {
//       t1 = imagebuf1.front()->header.stamp.toSec();
//       t2 = imagebuf2.front()->header.stamp.toSec();

//       lock2.lock();
//       while((t1-t2)>maxTimeDiff && imagebuf2.size()>1)
//       {
//         imagebuf2.pop();
//         t2 = imagebuf2.front()->header.stamp.toSec();
//       }
//       lock2.unlock();

//       lock1.lock();
//       while((t2-t1)>maxTimeDiff && imagebuf1.size()>1)
//       {
//         imagebuf1.pop();
//         t1 = imagebuf1.front()->header.stamp.toSec();
//       }
//       lock1.unlock();

//       if((t1-t2)>maxTimeDiff || (t2-t1)>maxTimeDiff)
//       {
//         std::cout << "big time difference" << std::endl;
//         return 0; 
//       }

//       lock1.lock();
//       image1 = GetImage(imagebuf1.front());
//       imagebuf1.pop();
//       lock1.unlock();

//       lock2.lock();
//       image2 = GetImage(imagebuf2.front());
//       imagebuf2.pop();
//       lock2.unlock();
//       return t1;
//     }
//     return 0;
// }


// void SyncSubscriber::depthCb(const ImageConstPtr& rgbImg, const ImageConstPtr& depth)
// {
//     mDepthMutex.lock();
//     img_rgb         = cv_bridge::toCvShare(rgbImg, "rgb8")->image.clone();
//     img_depth       = cv_bridge::toCvShare(depth, "16UC1")->image.clone(); //mono16/16UC1
//     mDepthMutex.unlock();
//     // static double initTime = rgbImg->header.stamp.toSec()*1e9;
//     double timestamp = rgbImg->header.stamp.toSec()*1e3;
//     // double timeDiff = timestamp-initTime;
//     my_sys->feed_rgbd_frame(img_rgb, img_depth, int64_t(timestamp));
// }

// void SyncSubscriber::depthCb(const ImageConstPtr& rgbImg,
//                             const ImageConstPtr& depth,
//                             const ImageConstPtr& maskDepth)
// {
//     ROS_INFO("got stereo data");
//     cv::Mat img_rgb   = cv_bridge::toCvShare(rgbImg, "rgb8")->image.clone();
//     cv::Mat img_depth = cv_bridge::toCvShare(depth, "16UC1")->image.clone();
//     cv::Mat l515MaskL = cv_bridge::toCvShare(maskDepth, "8UC1")->image.clone();
//     static double initTime = rgbImg->header.stamp.toSec()*1e9;
//     const int64_t timestamp = rgbImg->header.stamp.toSec()*1e9;
//     double timeDiff = timestamp-initTime;

//     my_sys->feed_rgbd_frame(img_rgb, img_depth, int64_t(timeDiff), l515MaskL);
// }


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