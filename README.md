# Real-time Semantic Reconstruction

This project is built for the perception system of an autonomous disinfection robot. The core
components of this module is in charge for SLAM and 3D semmantic reconstruction.

## Dependancy
* Build System
    * C++17
    * CMake >= 3.18  
      1. `pip install --upgrade cmake`
      2. `source ~/.bashrc`
    * CUDA >= 9.0
      1. `sudo apt install nvidia-cuda-toolkit`
* Externel Libraries
    * ~~[OpenVSLAM](https://github.com/xdspacelab/openvslam)
      (follow the
      [official installation guide](https://openvslam.readthedocs.io/en/master/installation.html)
      but with
      [this custom branch](https://github.com/alvinsunyixiao/openvslam/tree/system_inherit))~~
    * [ZED SDK](https://www.stereolabs.com/developers/release/) (optional)

## Note

For installing externel dependancies, to avoid collision with existing system packages, it is
recommended to install all packages (including the ones that OpenVSLAM depends on) into
a local directory such as `~/.local` instead of the system path `/usr/local`. This can
be easily achieved by adding `-DCMAKE_INSTALL_PREFIX=$HOME/.local` option to a CMake command.
Another benifit of this is that `sudo` privilege is not needed for the installation.

## Compilation

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Running SLAM

#### ZED 1

To run without ZED SDK (stereo calibration is required,
                        see example in `configs/zed_native_stereo.yaml`)

```bash
./run_zed_native -v <path_to_vocab.dbow2> -c ../configs/zed_native_stereo.yaml --devid 0
```

To run with ZED SDK (CUDA and ZED SDK is required)

```bash
./run_zed -v <path_to_vocab.dbow2> -c ../configs/zed_stereo.yaml
```

#### Intel SR300

```bash
./run_sr300 -v <path_to_vocab.dbow2> -c ../configs/sr300_rgbd.yaml --depth
```

## Running Reconstruction

This section is still WIP

## Running ROS
### installtion
1. [rviz_visual_tools](https://github.com/PickNikRobotics/rviz_visual_tools.git)
2. [TSDF2Mesh](https://github.com/yuzhou42/TSDF2Mesh.git)
### setup
- add libtorch
```
ln -s /path_to_torchlib third_party/libtorch
```
- setup dependencies
```
sudo apt-get install ros-noetic-eigen-stl-containers
sudo apt-get install ros-noetic-graph-msgs
```
- set model_path/calib_path/orb_vocab_path in launch/disinfslam.launch  

- we are only building the tsdf example, so comment the other examples out of `CMakeLists.txt`

- compile
```bash
catkin build  
```
### Running
```bash
roslaunch disinfslam disinfslam.launch 
roslaunch TSDF2Mesh tsdf2mesh.launch
```

### Visulization
```
rviz
# open recent config and selet ~/slam_ws/slam.rviz
```
## TODO

- Add logic to not update TSDF when tracking is lost
- Update installation.md
- Add a sciprt folder for easier user access
- Update segmentation inference example to take in custom images
- Add rotation arthimetics (e.g. rotation matrices interpolation)
- Use pre-processer to allow online.cc to compile without libtorch
- TensorRT for saving GPU memory
