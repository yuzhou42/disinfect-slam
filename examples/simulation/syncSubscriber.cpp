#include "syncSubscriber.h"

SyncSubsriber::SyncSubsriber()
{
    nh_.getParam("/simulate_disinf_slam/model_path", model_path);
    nh_.getParam("/simulate_disinf_slam/calib_path", calib_path);
    nh_.getParam("/simulate_disinf_slam/orb_vocab_path", orb_vocab_path);
    nh_.param("/simulate_disinf_slam/bbox_xy", bbox_xy, 4.0);
    nh_.param("/simulate_disinf_slam/renderer", renderFlag, false);
    nh_.param("/simulate_disinf_slam/use_mask", use_mask, false);

    image_transport::ImageTransport it(nh_);

    stereoLeft.subscribe(it, "/stereoLeft", 10);
    stereoRight.subscribe(it, "/stereoRight", 10);
    depth.subscribe(it, "/depth", 10);
    rgbImg.subscribe(it, "/rgbImg", 10);
    maskLeft.subscribe(it, "/maskLeft", 10);
    maskDepth.subscribe(it, "/maskDepth", 10);

    // publishers
    mPubTsdfGlobal = nh_.advertise<std_msgs::Float32MultiArray>("/tsdf_global", 4);
    mPubTsdfLocal  = nh_.advertise<std_msgs::Float32MultiArray>("/tsdf_local", 4);
    if (use_mask)
    {
        sync3_stereo_.reset(new Sync3(MySyncPolicy3(10), stereoLeft, stereoRight, maskLeft));
        sync3_stereo_->registerCallback(boost::bind(&SyncSubsriber::stereoCb, this, _1, _2, _3));

        sync3_depth_.reset(new Sync3(MySyncPolicy3(10), rgbImg, depth, maskDepth));
        sync3_depth_->registerCallback(boost::bind(&SyncSubsriber::depthCb, this, _1, _2, _3));
    }
    else
    {
        sync2_stereo_.reset(new Sync2(MySyncPolicy2(10), stereoLeft, stereoRight));
        sync2_stereo_->registerCallback(boost::bind(&SyncSubsriber::stereoCb, this, _1, _2));

        sync2_depth_.reset(new Sync2(MySyncPolicy2(10), rgbImg, depth));
        sync2_depth_->registerCallback(boost::bind(&SyncSubsriber::depthCb, this, _1, _2));
    }

    reconstTimer = nh_.createTimer(ros::Duration(0.2), &SyncSubsriber::reconstTimerCallback, this);
    poseTimer    = nh_.createTimer(ros::Duration(0.2), &SyncSubsriber::poseTimerCallback, this);

    // ROS_INFO_STREAM(calib_path);

    auto cfg = get_and_set_config(calib_path);
    my_sys   = std::make_shared<DISINFSystem>(calib_path, orb_vocab_path, model_path, renderFlag);
    mReconstrMsg.reset(new std_msgs::Float32MultiArray);
}

void SyncSubsriber::stereoCb(const ImageConstPtr& stereoLeft, const ImageConstPtr& stereoRight)
{
    ROS_INFO("got stereo data");
    cv::Mat img_left        = cv_bridge::toCvShare(stereoLeft, "bgr8")->image;
    cv::Mat img_right       = cv_bridge::toCvShare(stereoRight, "bgr8")->image;
    const int64_t timestamp = stereoLeft->header.stamp.toSec();
    my_sys->feed_stereo_frame(img_left, img_right, timestamp);
}

void SyncSubsriber::depthCb(const ImageConstPtr& rgbImg, const ImageConstPtr& depth)
{
    ROS_INFO("got depth data");
    cv::Mat img_rgb         = cv_bridge::toCvShare(rgbImg, "rgb8")->image;
    cv::Mat img_depth       = cv_bridge::toCvShare(depth, "16UC1")->image; //mono16/16UC1
    ROS_INFO("finished converting");
    const int64_t timestamp = rgbImg->header.stamp.toSec();
    my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp);
}

void SyncSubsriber::stereoCb(const ImageConstPtr& stereoLeft,
                             const ImageConstPtr& stereoRight,
                             const ImageConstPtr& maskLeft)
{
    ROS_INFO("got stereo data");
    cv::Mat img_left     = cv_bridge::toCvShare(stereoLeft, "bgr8")->image;
    cv::Mat img_right    = cv_bridge::toCvShare(stereoRight, "bgr8")->image;
    cv::Mat zedLeftMaskL = cv_bridge::toCvShare(maskLeft, "8UC1")->image;

    const int64_t timestamp = stereoLeft->header.stamp.toSec();
    my_sys->feed_stereo_frame(img_left, img_right, timestamp, zedLeftMaskL);
}

void SyncSubsriber::depthCb(const ImageConstPtr& rgbImg,
                            const ImageConstPtr& depth,
                            const ImageConstPtr& maskDepth)
{
    ROS_INFO("got stereo data");
    cv::Mat img_rgb   = cv_bridge::toCvShare(rgbImg, "rgb8")->image;
    cv::Mat img_depth = cv_bridge::toCvShare(depth, "16UC1")->image;
    cv::Mat l515MaskL = cv_bridge::toCvShare(maskDepth, "8UC1")->image;

    const int64_t timestamp = rgbImg->header.stamp.toSec();
    my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp, l515MaskL);
}

void SyncSubsriber::SyncSubsriber::reconstTimerCallback(const ros::TimerEvent&)
{
    static unsigned int last_query_time = 0;
    static size_t last_query_amount     = 0;
    // static float bbox = 4.0;
    static float x_range[2]           = {-bbox_xy, bbox_xy};
    static float y_range[2]           = {-bbox_xy, bbox_xy};
    static float z_range[2]           = {-2.0, 4.0};
    static BoundingCube<float> volumn = {
        x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]};
    static ros::Time stamp;
    uint32_t tsdf_global_sub = mPubTsdfGlobal.getNumSubscribers();
    uint32_t tsdf_local_sub  = mPubTsdfLocal.getNumSubscribers();

    if (tsdf_global_sub > 0)
    {
        const auto st          = get_timestamp<std::chrono::milliseconds>(); // nsec
        auto mSemanticReconstr = my_sys->query_tsdf(volumn);
        const auto end         = get_timestamp<std::chrono::milliseconds>();
        last_query_time        = end - st;
        last_query_amount      = mSemanticReconstr.size();
        std::cout << "Last queried %lu voxels " << last_query_amount << ", took " << last_query_time
                  << " ms" << std::endl;
        int size = last_query_amount * sizeof(VoxelSpatialTSDF);
        mReconstrMsg->data.resize(size);
        std::memcpy(&(mReconstrMsg->data[0]),
                    (char*)mSemanticReconstr.data(),
                    last_query_amount * sizeof(VoxelSpatialTSDF));
        mPubTsdfGlobal.publish(mReconstrMsg);
    }

    if (tsdf_local_sub > 0)
    {
        const auto st = get_timestamp<std::chrono::milliseconds>(); // nsec
        float x_off   = transformStamped.transform.translation.x,
              y_off   = transformStamped.transform.translation.y,
              z_off   = transformStamped.transform.translation.z;
        std::cout << "x_off: " << x_off << "  y_off: " << y_off << "  z_off: " << z_off
                  << std::endl;
        BoundingCube<float> volumn_local = {x_off + x_range[0],
                                            x_off + x_range[1],
                                            y_off + y_range[0],
                                            y_off + y_range[1],
                                            z_off + z_range[0],
                                            z_off + z_range[1]};
        auto mSemanticReconstr           = my_sys->query_tsdf(volumn_local);
        const auto end                   = get_timestamp<std::chrono::milliseconds>();
        last_query_time                  = end - st;
        last_query_amount                = mSemanticReconstr.size();
        std::cout << "Last queried %lu voxels " << last_query_amount << ", took " << last_query_time
                  << " ms" << std::endl;
        int size = last_query_amount * sizeof(VoxelSpatialTSDF);
        mReconstrMsg->data.resize(size);
        std::memcpy(&(mReconstrMsg->data[0]),
                    (char*)mSemanticReconstr.data(),
                    last_query_amount * sizeof(VoxelSpatialTSDF));
        mPubTsdfLocal.publish(mReconstrMsg);
    }
}

void SyncSubsriber::poseTimerCallback(const ros::TimerEvent&)
{
    static tf2_ros::TransformBroadcaster mTfSlam;
    static ros::Time stamp;
    static ros::Rate rate(30);
    u_int64_t t_query    = get_timestamp<std::chrono::milliseconds>();
    stamp.sec            = t_query / 1000;                 // s
    stamp.nsec           = (t_query % 1000) * 1000 * 1000; // ns
    SE3<float> mSlamPose = my_sys->query_camera_pose(t_query);
    // pubPose(stamp, mSlamPose, mTfSlam);
    tf2::Transform tf2_trans;
    tf2::Transform tf2_trans_inv;

    tf2::Matrix3x3 R(mSlamPose.m00,
                     mSlamPose.m01,
                     mSlamPose.m02,
                     mSlamPose.m10,
                     mSlamPose.m11,
                     mSlamPose.m12,
                     mSlamPose.m20,
                     mSlamPose.m21,
                     mSlamPose.m22);
    tf2::Vector3 T(mSlamPose.m03, mSlamPose.m13, mSlamPose.m23);

    tf2_trans.setBasis(R);
    tf2_trans.setOrigin(T);

    tf2_trans_inv = tf2_trans.inverse();

    transformStamped.header.stamp    = stamp;
    transformStamped.header.frame_id = "slam";
    transformStamped.child_frame_id  = "zed";

    tf2::Quaternion q                     = tf2_trans_inv.getRotation();
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();

    tf2::Vector3 t                           = tf2_trans_inv.getOrigin();
    transformStamped.transform.translation.x = t[0];
    transformStamped.transform.translation.y = t[1];
    transformStamped.transform.translation.z = t[2];

    mTfSlam.sendTransform(transformStamped);
}