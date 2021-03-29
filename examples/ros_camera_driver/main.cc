#include "ros_interface.h"

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "ros_disinf_slam");
    ros::NodeHandle nh;
    RosInterface rosInterface;
    ros::spin();
}
