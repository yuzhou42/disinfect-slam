#include "ros_interface.h"

RosInterface::RosInterface():initT(false)
{
    mNh.getParam("/ros_disinf_slam/model_path", model_path); 
    mNh.getParam("/ros_disinf_slam/calib_path", calib_path); 
    mNh.getParam("/ros_disinf_slam/orb_vocab_path", orb_vocab_path); 
    mNh.param("/ros_disinf_slam/devid", devid, 2);
    mNh.param("/ros_disinf_slam/bbox_xy", bbox_xy, 4.0);
    mNh.param("/ros_disinf_slam/renderer", renderFlag, false);
    mNh.param("/ros_disinf_slam/global_mesh", global_mesh, true);
    mNh.param("/ros_disinf_slam/require_mesh", require_mesh, true);

    auto cfg = GetAndSetConfig(calib_path);
    zed_native.reset(new ZEDNative(*cfg, devid));
    l515.reset(new L515());
    image_transport::ImageTransport it(mNh);
    maskLeft = it.subscribe("/maskLeft", 1, &RosInterface::zedMaskCb, this);
    maskDepth = it.subscribe("/maskDepth", 1, &RosInterface::l515MaskCb, this);
    meshPub = mNh.advertise<shape_msgs::Mesh>("/mesh", 1);
    mPubL515RGB = mNh.advertise<sensor_msgs::Image>("/l515_rgb", 1);
    mPubL515Depth = mNh.advertise<sensor_msgs::Image>("/l515_depth", 1);
    mPubZEDImgL = mNh.advertise<sensor_msgs::Image>("/zed_left_rgb", 1);
    mPubZEDImgR = mNh.advertise<sensor_msgs::Image>("/zed_right_rgb", 1);
    // cv bridges
    mL515RGBBrg.reset(new cv_bridge::CvImage);
    mL515DepthBrg.reset(new cv_bridge::CvImage);
    mZEDImgLBrg.reset(new cv_bridge::CvImage);
    mZEDImgRBrg.reset(new cv_bridge::CvImage);

    my_sys   = std::make_shared<DISINFSystem>(calib_path, orb_vocab_path, model_path, renderFlag);
    mReconstrMsg.reset(new std_msgs::Float32MultiArray);
    visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("world","/mesh_visual", mNh));
    visual_tools_->setPsychedelicMode(false);
    visual_tools_->loadMarkerPub();
    tfListener= std::make_shared<tf2_ros::TransformListener>(tfBuffer);

    // reconstTimer = mNh.createTimer(ros::Duration(0.2), &RosInterface::reconstTimerCallback, this);
    // poseTimer    = mNh.createTimer(ros::Duration(0.05), &RosInterface::poseTimerCallback, this);
    run();
}

void RosInterface::publishImage(cv_bridge::CvImagePtr & imgBrgPtr, const cv::Mat & img, ros::Publisher & pubImg, std::string imgFrameId, std::string dataType, ros::Time t){
  imgBrgPtr->header.stamp = t;
  imgBrgPtr->header.frame_id = imgFrameId;
  imgBrgPtr->encoding = dataType;
  imgBrgPtr->image = img;
  pubImg.publish(imgBrgPtr->toImageMsg());
}

void RosInterface::tsdfCb()
{
    static Eigen::Isometry3d pose;
    if(!initT){
        try{
            geometry_msgs::TransformStamped transformStampedInit = tfBuffer.lookupTransform("world", "slam",
                                    ros::Time(0));
            pose = tf2::transformToEigen(transformStampedInit);
            std::cout<<"Init world slam transform"<<std::endl;
            initT = true;

        }
        catch (tf2::TransformException &ex) {
            ROS_WARN("%s",ex.what());   
        }
    }

    auto values = mReconstrMsg->data;
    // ROS_INFO("I heard tsdf of size: ", msg->data.size());
    const auto st = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());

    int numPoints = mReconstrMsg->data.size()/4;
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

    // MergeVertices(mesh, 0.05);

    int vertsSize = mesh.verts.size();
    int trisSize = mesh.tris.size();

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

void RosInterface::run() {

  std::thread t_slam([&]() {
    std::cout<<"Start SLAM thread!"<<std::endl;
    cv::Mat img_left, img_right, zedLeftMaskL;
    ros::Time ros_stamp;
    while (ros::ok()) {
      const int64_t timestamp = zed_native->GetStereoFrame(&img_left, &img_right);
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
    std::cout<<"Start TSDF thread!"<<std::endl;
    cv::Mat img_rgb, img_depth, l515MaskL;
    ros::Time ros_stamp;
    while (ros::ok()) {
      const int64_t timestamp = l515->GetRGBDFrame(&img_rgb, &img_depth);
        mask_lock.lock();
        l515MaskL = l515Mask.clone();
        mask_lock.unlock();
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
      std::cout<<"Start Reconstruction thread!"<<std::endl;
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
        //   const auto st = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
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

      tsdfCb();
      ros::spinOnce();
      rate.sleep();
    }
  });

  std::thread t_pose([&](){
    std::cout<<"Start Pose thread!"<<std::endl;
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

void RosInterface::zedMaskCb(const sensor_msgs::ImageConstPtr& msg)
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

void RosInterface::l515MaskCb(const sensor_msgs::ImageConstPtr& msg)
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
 
