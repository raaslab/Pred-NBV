# Pred-NBV

## Running Pred-NBV in AirSim Simulation
Required Simulation Packages
- AirSim
- MoveIt
- [PoinTr](https://github.com/yuxumin/PoinTr)

Start AirSim in Unreal Engine environment

Run AirSim ROS Wrapper in 1st terminal
```
roslaunch airsim_ros_pkgs airsim_node.launch
```

Run depth image segmentation in 2nd terminal
```
rosrun airsim_moveit_navigation depth_segmentation
```
Run depth to pointcloud in 3rd terminal
```
roslaunch depth_image_proc point_cloud_xyz_radial.launch
```
Run PoinTr to AirSim TF in 4th terminal
```
rosrun tf2_ros static_transform_publisher 0 0 0 0.7071068 0 0 -0.7071068 world_ned pointr
```
Launch MoveIt configuration in 5th terminal
```
roslaunch airsim_moveit_config gatsbi.launch
```
Run AirSim + MoveIt Client in 6th terminal
```
rosrun airsim_moveit_navigation airsim_navigator
```
Run Pred-NBV Reconstruction Pipeline in 7th terminal
```
rosrun airsim_moveit_navigation pointr_reconstruction.py
```