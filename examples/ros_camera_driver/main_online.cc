#include "ros_online.h"

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "ros_online_orb3");
    RosInterface rosInterface;
    ros::spin();
}
