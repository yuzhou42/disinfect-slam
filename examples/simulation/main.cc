#include "syncSubscriber.h"

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "simulation_disinf_slam");
    SyncSubsriber syncSubsriber;
    ros::spin();
}
