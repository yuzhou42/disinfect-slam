cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11/bin/nvcc)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 61 72 75)

project(disinfslam LANGUAGES CXX C CUDA)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

# set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_ARCHITECTURES 61 72 75)

SET(KLAMPT_ROOT "$ENV{HOME}/Klampt" CACHE PATH "Klampt root directory")
if ( "${KLAMPT_ROOT}" )
  SET(KLAMT_ROOT "${KLAMT_ROOT}")
endif()
SET (CMAKE_MODULE_PATH "${KLAMPT_ROOT}/CMakeModules")
MESSAGE(STATUS "Looking for FindKlampt.cmake in ${CMAKE_MODULE_PATH}")
# COMPILER SETTINGS (default: Release) and flags
INCLUDE(CompilerSettings)
FIND_PACKAGE(Klampt REQUIRED)
ADD_DEFINITIONS(${KLAMPT_DEFINITIONS})
INCLUDE_DIRECTORIES(${KLAMPT_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_PREFIX_PATH "/home/yu/.local2")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    geometry_msgs
    nodelet
    sensor_msgs
    std_msgs
    tf2_geometry_msgs
    tf2_ros
    cv_bridge
    shape_msgs
    image_transport
    rviz_visual_tools
)

# OpenGL
find_package(GLEW REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
# Slam
find_package(Eigen3 EXACT 3.3.91)
# find_package(openvslam REQUIRED)
find_package(Pangolin REQUIRED) # TODO(alvin): to be completely replaced by Dear ImGUI
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
# Zed SDK (optional)
find_package(ZED 3 REQUIRED)
# Camera SDKs
find_package(realsense2 REQUIRED)
# find_package(OpenCV REQUIRED)
# DL Inference
find_package(Torch QUIET HINTS ${CMAKE_SOURCE_DIR}/third_party/libtorch)
if (NOT Torch_FOUND)
    MESSAGE(WARNING "libtorch not found. Excluding DL inference engine and examples.")
endif()

# find_package(ORB_SLAM3   EXACT PATHS /home/yu/.local3/share/eigen3 )

# enable testing
# enable_testing()
# include(GoogleTest)

# Compile Flags
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O2 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O2 -march=native")

catkin_package()

set(ORB_DIR "$ENV{HOME}/ORB_SLAM3")
if ( "${ORB_DIR}" )
    set(ORB_DIR "${ORB_DIR}")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${OpenGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${KLAMPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${ORB_DIR}
    ${ORB_DIR}/include
    ${ORB_DIR}/include/CameraModels  
)

set(ORB_SLAM3_LIBS 
    ${OpenCV_LIBS} 
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${ORB_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${ORB_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${ORB_DIR}/lib/libORB_SLAM3.so
    -lboost_system
    -lboost_serialization
    -lcrypto
)
# Interface wrapper lib for OpenVSLAM
# add_library(openvslam INTERFACE)
# target_link_libraries(openvslam
#     INTERFACE popl openvslam::openvslam openvslam::pangolin_viewer)

add_subdirectory(third_party)
add_subdirectory(utils)
add_subdirectory(cameras)
add_subdirectory(modules)
# add_subdirectory(${ORB_DIR} ${ORB_DIR}+"/build2")

if (TORCH_FOUND)
    add_subdirectory(segmentation)
endif ()
add_subdirectory(disinfect_slam)
add_subdirectory(examples)

# -------------------------------------------------------------------- #
# ----------------------- clang format target ------------------------ #
# -------------------------------------------------------------------- #

find_program(CLANG_FORMAT_EXE NAMES
    clang-format-10
    clang-format-9
    clang-format-8
    clang-format)

# gather all source clode
file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/*.cc
    ${CMAKE_SOURCE_DIR}/*.h
    ${CMAKE_SOURCE_DIR}/*.hpp
    ${CMAKE_SOURCE_DIR}/*.cu
    ${CMAKE_SOURCE_DIR}/*.cuh
)

# exclude some directories
list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX .*/.*build.*/.*)
list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX .*/third_party/.*)

# create formatting helper targets
if (CLANG_FORMAT_EXE)
    set(RUN_CLANG_FORMAT ${CMAKE_SOURCE_DIR}/run-clang-format.py)
    # format code in placeEigen
    add_custom_target(format
        COMMAND python3 ${RUN_CLANG_FORMAT} --clang-format-executable ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
        DEPENDS ${RUN_CLANG_FORMAT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    # check for format violations
    add_custom_target(check-format
        COMMAND python3 ${RUN_CLANG_FORMAT} --clang-format-executable ${CLANG_FORMAT_EXE} ${ALL_SOURCE_FILES}
        DEPENDS ${RUN_CLANG_FORMAT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif (CLANG_FORMAT_EXE)
