import numpy as np
import open3d as o3d
import copy

from pose_to_image import *
from nbv_simulator import *

valid_candidate_poses = np.load('/home/user/valid_candidate_poses.npy')
trajectories = np.load('/home/user/rrt_trajectories.npy')

nbv_planner = Pose2Image()
observed_pcl = []
next_best_pcd = []
pred_pcd_list = []
observed_dimg = []
cam_location_list = []
global_best_pos = np.array([trajectories[0][1], trajectories[0][2], trajectories[0][3]])
print(global_best_pos)
pred_pcd = None

def NBV2D_Circular(pred_pcd, diff_ids):

    global valid_candidate_poses
    candidate_poses = valid_candidate_poses

    proj_img_list = []
    proj_color_list = []
    num_pts_list = []
    pose_list = []
    pcd_list = []
    proj_depth_img_list = []
    hidden_ids_list = []

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))

    for pose in candidate_poses:
        pcd = copy.copy(pred_pcd)

        # Left rotation is +ve
        #         print(pose)

        yaw = -90 - np.degrees(np.arctan2(pose[2], pose[0]))

        proj_img_pts, proj_img_color, proj_pcd, hid_ids = nbv_planner.get_image(pcd, x=pose[0], y=pose[1], z=pose[2],
                                                                                roll=0, pitch=0, yaw=yaw,
                                                                                diff_ids=diff_ids)

        proj_depth_img = nbv_planner.depth_image_for_pcd(proj_pcd)

        proj_img_list.append(proj_img_pts)
        proj_color_list.append(proj_img_color)
        num_pts_list.append(len(proj_img_pts))
        pcd_list.append(proj_pcd)
        proj_depth_img_list.append(proj_depth_img)
        hidden_ids_list.append(copy.copy(hid_ids))

        del pcd

    return proj_img_list, proj_color_list, num_pts_list, candidate_poses, pcd_list, proj_depth_img_list, hidden_ids_list


input_data = np.load('/home/user/airsim_input_before_prediction.npy')
pcd_partial = o3d.geometry.PointCloud()
pcd_partial.points = o3d.utility.Vector3dVector(input_data)

full_pc = np.load('/home/user/dense.npy')
full_pc = full_pc.reshape((full_pc.shape[1], full_pc.shape[2]))

partial_pc = np.load('/home/user/partial.npy')
partial_pc = partial_pc.reshape((partial_pc.shape[1], partial_pc.shape[2]))

#predicted_data = full_pc
#for item in partial_pc:
#        idx = np.argwhere(full_pc == item)
#        predicted_data = np.delete(predicted_data, idx[0][0], axis=0)

#partial_pc_pcd = o3d.geometry.PointCloud()
#partial_pc_pcd.points = o3d.utility.Vector3dVector(partial_pc)

#o3d.visualization.draw_geometries([partial_pc_pcd])

# Convert to open3d point cloud
pred_pcd1 = o3d.geometry.PointCloud()
pred_pcd1.points = o3d.utility.Vector3dVector(full_pc)

pred_pcd = apply_icp(source=copy.deepcopy(pred_pcd1), target=copy.deepcopy(pcd_partial), voxel_size=0.01)
# pred_pcd = pred_pcd1
pred_pcd_list.append(copy.deepcopy(pred_pcd))

#     pred_pcd = reorient(pred_pcd,x=x_coord, y=0, z=z_coord, rx=0, ry=z_rot, rz=0)

#     observed_pcl.append(reorient(pcd_partial, x=x_coord, y=0, z=z_coord, rx=0, ry=z_rot, rz=0))

cam_location_list.append(global_best_pos[None, :])
observed_pcl.append(copy.deepcopy(pcd_partial))

### Running NBV planner on it to get the best new pose
#     diff_pcd = nbv_planner.remove_points_from_1(pred_pcd, pcd_partial, thresh=1e-5)
#     diff_pcd = orient(diff_pcd, x=x_coord, y=0, z=z_coord, rx=0, ry=z_rot, rz=0)


## getting IDs for new points
diff_ids = nbv_planner.get_nonoverlapping_points(pcd1=pred_pcd, pcd2=pcd_partial, thresh=1e-2)
## Reorienting to the camera frame because we will find NBV relatively
# pred_pcd = orient(pred_pcd, x=x_coord, y=0, z=z_coord, rx=0, ry=z_rot, rz=0)

#     break

#### NBV 2D
#     proj_img_list, proj_color_list, num_pts_list, pose_list, pcd_list = NBV2D(pred_pcd, diff_ids)
#     proj_img_list, proj_color_list, num_pts_list, pose_list, pcd_list = NBV3D(pred_pcd, diff_ids)

# pred_pcd = reorient(pred_pcd, x=x_coord, y=0, z=z_coord, rx=0, ry=z_rot, rz=0)
proj_img_list, proj_color_list, num_pts_list, pose_list, pcd_list, proj_depth_img_list, proj_hid_id_list = NBV2D_Circular(pred_pcd, diff_ids)
####

distances = np.zeros(pose_list.shape[0])
for i in range(0, pose_list.shape[0]):
    candidate_trajectory = trajectories[trajectories[:, 0] == i, :]
    candidate_trajectory = np.delete(candidate_trajectory, 0, 1)

    distances[i] = np.sum(np.linalg.norm((candidate_trajectory[1:] - candidate_trajectory[0:-1]), axis=1))

distances = np.linalg.norm((pose_list - global_best_pos), axis=1)

#     best_pos = pose_list[np.argmax(num_pts_list)]
num_pts_pct = np.asarray(num_pts_list)
num_pts_pct = num_pts_pct / (num_pts_pct.max() + 1)

search_order = np.argsort(distances)
for i_s in search_order:
    if distances[i_s] > 1e-3 and num_pts_pct[i_s] > 0.90:
        break
best_pos = pose_list[i_s]

np.save('/home/user/pose_list.npy', pose_list)
print(f'Best pose: {best_pos}')
global_best_pos = best_pos
global_best_yaw = - 90 - np.degrees(np.arctan2(global_best_pos[2], global_best_pos[0]))
np.save('/home/user/next_candidate.npy', global_best_pos)
print(f'Global pose: {global_best_pos}, {global_best_yaw}')
print(f'Distance: {distances[i_s]}, %vis: {num_pts_pct[i_s]}')
print()