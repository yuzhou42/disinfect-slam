add_library(disinf_slam_top_level disinfect_slam.cc)
target_link_libraries(disinf_slam_top_level
    PUBLIC zed l515 gl_utils  tsdf_module  popl segmentation rotation_math ${OpenCV_LIBS}  ${ORB_SLAM3_LIBS})
target_include_directories(disinf_slam_top_level PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})