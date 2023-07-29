# Vision-based NMPC for target tracking using UAVs
This is a ROS C++ package of various vison-based Nonlinear Model Predictive Control (NMPC) strategies for UAV aiming on target tracking. The general application of these controllers is a dynamic coastline in the presenc of waves.
The different version of visual servoing will be analyzed below.
The controllers were all developed in a UAV synthetic simulation environment: https://github.com/sotomotocross/UAV_simulator_ArduCopter.git

## Event-triggered classic IBVS-NMPC method for target tracking
We initially implemented an event-triggered IBVS-NMPC strategy for an underactuated UAV aiming target tracking which in our case is a constantly moving coastline in the presence of waves.
Initially we utilized a classic IBVS-NMPC for target tracking:
```
$ roslaunch vsc_nmpc_uav_target_tracking track_ibvs_nmpc.launch
```
and then we also developed an event-triggered version of the controller
```
$ roslaunch vsc_nmpc_uav_target_tracking et_track_ibvs_nmpc.launch
```
This is the core of publication [[1]](#1).

## Image-moments like NMPC for target tracking

```
$ roslaunch vsc_nmpc_uav_target_tracking ibvs_img_moments_coast_tracking.launch
```

```
$ roslaunch vsc_nmpc_uav_target_tracking nmpc_img_moments_coast_tracking.launch
```

```
$ roslaunch vsc_nmpc_uav_target_tracking nmpc_img_moments_coast_tracking_with_cbf.launch
```

## Event-triggered image-moments like NMPC for target tracking

```
$ roslaunch vsc_nmpc_uav_target_tracking event_triggered_img_mom_nmpc_no_tracker.launch
```

```
$ roslaunch vsc_nmpc_uav_target_tracking event_triggered_img_mom_nmpc.launch
```



## References
<a id="1">[1]</a> 
S. N. Aspragkathos, M. Sinani, G. C. Karras, and K. J. Kyriakopoulos, “An Event-triggered Visual Servoing Predictive Control Strategy for the Surveillance of Contour-based Areas using Multirotor Aerial Vehicles", in IEEE/RSJ 2022 International Conference on Intelligent Robots and Systems (IROS), IEEE, 2022, pp. 375–381, IEEE, 2022, [10.1109/IROS47612.2022.9981176](10.1109/IROS47612.2022.9981176)