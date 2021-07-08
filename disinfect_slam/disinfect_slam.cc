#include "disinfect_slam/disinfect_slam.h"

DISINFSystem::DISINFSystem(std::string camera_config_path,
                           std::string vocab_path,
                           std::string seg_model_path,
                           const ORB_SLAM3::System::eSensor sensor, 
                           bool pangolin_view)
{
    SLAM_ = std::make_shared<ORB_SLAM3::System>(
        vocab_path, camera_config_path, sensor, pangolin_view);

    //   SEG_ = std::make_shared<inference_engine>(seg_model_path);
    TSDF_ = std::make_shared<TSDFSystem>(0.05,
                                         0.2,
                                         4,
                                         GetIntrinsicsFromFile(camera_config_path),
                                         GetExtrinsicsFromFile(camera_config_path));

    depthmap_factor_ = GetDepthFactorFromFile(camera_config_path);
    camera_pose_manager_ = std::make_shared<pose_manager>();

    // if (rendering_flag) {
    //   RENDERER_ = std::make_shared<ImageRenderer>("tsdf", SLAM_, TSDF_, camera_config_path);
    // }
}

DISINFSystem::~DISINFSystem() {  }

// void DISINFSystem::run() { RENDERER_->Run(); }

void DISINFSystem::feed_rgbd_frame(const cv::Mat& img_rgb,
                                   const cv::Mat& img_depth,
                                   int64_t timestamp,
                                   const cv::Mat& mask)
{
    cv::Mat my_img_rgb, my_img_depth, my_mask; // local Mat that will be modified
    const SE3<float> posecam_P_world = camera_pose_manager_->query_pose(timestamp);
    cv::resize(img_rgb, my_img_rgb, cv::Size(), .5, .5);
    cv::resize(img_depth, my_img_depth, cv::Size(), .5, .5);
    my_img_depth.convertTo(my_img_depth, CV_32FC1,
                           1. / depthmap_factor_); // depth scale
    if (!mask.empty())
    {
        // std::cout<<"mask is not empty!"<<std::endl;
        cv::resize(mask, my_mask, cv::Size(), .5, .5);
        cv::Size s   = my_img_depth.size();
        int num_rows = s.height;
        int num_cols = s.width;
        int cnt      = 0;
        for (unsigned int i = 0; i < num_rows; ++i)
        {
            for (unsigned int j = 0; j < num_cols; ++j)
            {
                if (my_mask.at<unsigned char>(i, j) == 0)
                {
                    my_img_depth.at<float>(i, j) = 0.0;
                    cnt++;
                }
            }
        }
        // std::cout<<"mask count: "<<cnt<<std::endl;
        // cv::imshow("mask_depth", my_img_depth);
        // cv::waitKey(1);
    }
    //   std::vector<cv::Mat> prob_map = SEG_->infer_one(my_img_rgb, false);
    TSDF_->Integrate(posecam_P_world, my_img_rgb, my_img_depth);
}

void DISINFSystem::feed_stereo(const cv::Mat& img_left,
                                   const cv::Mat& img_right,
                                   double timestamp)
{
    cv::Mat Tcw(4, 4, CV_32FC1);
    if(!Tcw.empty()){
        Eigen::Matrix<float, 4, 4> eigenT;
        cv::cv2eigen(Tcw, eigenT);
        const SE3<float> posecam_P_world(eigenT);
        camera_pose_manager_->register_valid_pose(timestamp*1e3, posecam_P_world); //s to ms

    }
}

void DISINFSystem::feed_stereo_IMU(const cv::Mat& img_left,
                                   const cv::Mat& img_right,
                                   double timestamp,
                                   const std::vector<ORB_SLAM3::IMU::Point>& vImuMeas)
{
    cv::Mat Tcw(4, 4, CV_32FC1);
    Tcw = SLAM_->TrackStereo(img_left, img_right, timestamp, vImuMeas);
    if(!Tcw.empty()){
        Eigen::Matrix<float, 4, 4> eigenT;
        cv::cv2eigen(Tcw, eigenT);
        const SE3<float> posecam_P_world(eigenT);
        camera_pose_manager_->register_valid_pose(timestamp*1e3, posecam_P_world); //s to ms

    }
   
}

SE3<float> DISINFSystem::query_camera_pose(const int64_t timestamp)
{
    return this->camera_pose_manager_->query_pose(timestamp);
}

std::vector<VoxelSpatialTSDF> DISINFSystem::query_tsdf(const BoundingCube<float>& volumn)
{
    return TSDF_->Query(volumn);
}
