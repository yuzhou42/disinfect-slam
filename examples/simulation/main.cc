#include "syncSubscriber.h"

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "simulation_disinf_slam");
    ros::NodeHandle nh;
    SyncSubsriber syncSubsriber;
    ros::spin();
}
