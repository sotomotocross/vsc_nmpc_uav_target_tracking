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

## Image-moments like NMPC for target tracking (study under review)
The proposed image moments-based Visual Servoing NMPC scheme is designed for surveillance and tracking of arbitrary evolving shapes. The tracking task is formulated based on moment-like quantities (centroid, area, orientation) in order to effectively track such shapes. Furthermore, we develop the dynamic model of the aforementioned quantities and propose a VS-NMPC scheme according to the aforementioned model. Additionally, the proposed controller incorporates an estimation term of the motion of the target to enable more efficient tracking.
```
$ roslaunch vsc_nmpc_uav_target_tracking nmpc_img_moments_coast_tracking.launch
```
The approach utilizes NMPC to handle input and state constraints and employs a safety-critical NMPC strategy incorporating barrier functions to ensure system safety, through visibility and state constraints, as well as optimal performance. In order to guarantee that the target is maintained within the camera FoV, the states, i.e. its coordinates on the image plane, should not violate a set of constraints. The visibility and state constraints are derived through the incorporation in BFs aiming for the visual servo control to generate the control inputs satisfying them.
```
$ roslaunch vsc_nmpc_uav_target_tracking nmpc_img_moments_coast_tracking_with_cbf.launch
```
Also an image-moments like visual servoing implementation has been formulated to compare the two methodologies.
```
$ roslaunch vsc_nmpc_uav_target_tracking ibvs_img_moments_coast_tracking.launch
```
This is the core of an under submission/review publication.

## Event-triggered image-moments like NMPC for target tracking (study under review)
This study introduces an aperiodic vision-based NMPC with a triggering mechanism for autonomous visual surveillance to reduce computational effort and energy consumption. The dynamic model of the system aiming to track contour-based areas featuring evolving features, which is propagated in the NMPC scheme, is extracted through the utilization of image moments. Also, the proposed strategy incorporates target motion estimation term through a hybrid model-based/data-driven (MB/DD) vision-based framework. The proposed scheme effectively handles visibility, input constraints, and disturbances from the environment.
```
$ roslaunch vsc_nmpc_uav_target_tracking event_triggered_img_mom_nmpc_no_tracker.launch
$ roslaunch vsc_nmpc_uav_target_tracking event_triggered_img_mom_nmpc.launch

```
This is the core of an under submission/review publication.


## References
<a id="1">[1]</a> 
S. N. Aspragkathos, M. Sinani, G. C. Karras, and K. J. Kyriakopoulos, “An Event-triggered Visual Servoing Predictive Control Strategy for the Surveillance of Contour-based Areas using Multirotor Aerial Vehicles", in IEEE/RSJ 2022 International Conference on Intelligent Robots and Systems (IROS), IEEE, 2022, pp. 375–381, IEEE, 2022, [10.1109/IROS47612.2022.9981176](10.1109/IROS47612.2022.9981176)
<a id="2">[2]</a> 
S. N. Aspragkathos, G. C. Karras and K. J. Kyriakopoulos, "Event-Triggered Image Moments Predictive Control for Tracking Evolving Features Using UAVs," in IEEE Robotics and Automation Letters, vol. 9, no. 2, pp. 1019-1026, Feb. 2024, doi: 10.1109/LRA.2023.3339064.)
