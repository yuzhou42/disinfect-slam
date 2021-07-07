#include "ros_offline.h"

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "ros_offline_orb3");
    SyncSubscriber syncSubscriber;
    ros::spin();
}
