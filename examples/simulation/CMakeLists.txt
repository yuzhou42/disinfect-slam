# From parent CMake
# It's safe to assume that libtorch is installed.
if (CMAKE_CUDA_COMPILER)
    add_executable(simulate_disinf_slam main.cc)
    target_link_libraries(simulate_disinf_slam PUBLIC ${catkin_LIBRARIES}
        PRIVATE disinf_slam_top_level popl openvslam ${OpenCV_LIBS})
    add_dependencies(simulate_disinf_slam disinf_slam_top_level)
    target_include_directories(simulate_disinf_slam PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif ()
