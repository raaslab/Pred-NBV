import numpy as np
import open3d as o3d

input = np.load('/home/user/dense.npy')
input = input.reshape((input.shape[1], input.shape[2]))
centroid = input.mean(axis=0)
max_dist = np.linalg.norm(input - centroid, axis=1).max()

mid_radius = max_dist * 1.5
top_radius = mid_radius * 0.8
bottom_radius = mid_radius * 0.8

max_height = np.max(input[:, 1])
min_height = np.min(input[:, 1])
height_range = np.ptp(input[:, 1])

shift = mid_radius * np.cos(np.radians(45))

candidate_poses = np.array(
    [[mid_radius * np.cos(np.radians(ang)), 0, mid_radius * np.sin(np.radians(ang))] for ang in range(0, 360, 30)])
candidate_poses[:, 1] = 0

candidate_poses = np.concatenate((candidate_poses, np.array(
    [[top_radius * np.cos(np.radians(ang)), max_height/2, top_radius * np.sin(np.radians(ang))] for ang in range(0, 360, 30)])), axis=0)

candidate_poses = np.concatenate((candidate_poses, np.array(
    [[bottom_radius * np.cos(np.radians(ang)), -max_height/2, bottom_radius * np.sin(np.radians(ang))] for ang in range(0, 360, 30)])), axis=0)

candidate_poses += centroid


np.save('/home/user/candidate_poses.npy', candidate_poses)

'''
input = np.concatenate((input, candidate_poses), axis=0)
input_data_pcd = o3d.geometry.PointCloud()
input_data_pcd.points = o3d.utility.Vector3dVector(input)

o3d.visualization.draw_geometries([input_data_pcd])
'''

print('Done')