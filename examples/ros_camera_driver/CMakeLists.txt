# From parent CMake
# It's safe to assume that libtorch is installed.
if (CMAKE_CUDA_COMPILER)
    # add_executable(ros_disinf_slam main.cc ros_interface.cpp)
    # target_link_libraries(ros_disinf_slam PUBLIC ${catkin_LIBRARIES} ${KLAMPT_LIBRARIES}
    #     PRIVATE disinf_slam_top_level popl openvslam zed_native l515 ${OpenCV_LIBS})
    # add_dependencies(ros_disinf_slam disinf_slam_top_level)
    # target_include_directories(ros_disinf_slam PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # add_executable(ros_online_orb3 main_online.cc ros_online.cc)
    # target_link_libraries(ros_online_orb3 PUBLIC ${catkin_LIBRARIES} ${KLAMPT_LIBRARIES} ${ORB_SLAM3_LIBS}
    #     PRIVATE disinf_slam_top_level popl zed l515 ${OpenCV_LIBS})
    # add_dependencies(ros_online_orb3 disinf_slam_top_level)
    # target_include_directories(ros_online_orb3 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    add_executable(ros_offline_orb3 main_offline.cc ros_offline.cc)
    target_link_libraries(ros_offline_orb3 PUBLIC ${catkin_LIBRARIES} ${KLAMPT_LIBRARIES} ${ORB_SLAM3_LIBS}
         disinf_slam_top_level)
    add_dependencies(ros_offline_orb3 disinf_slam_top_level) # means disinf_slam_top_level should be generated first
    target_include_directories(ros_offline_orb3 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


endif ()


