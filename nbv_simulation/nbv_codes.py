def NBV3D(pcd, diff_ids):
    im_height, im_width = 480, 480
    intrinsic = np.array([[im_width//2, 0, im_width//2],
                          [0, im_height//2, im_height//2],
                          [0, 0, 1]])
    
    radius = 0.1
    angle = 30
    
    x_shift = radius * np.cos(np.radians(angle))
    z_shift = radius * np.cos(np.radians(angle))
    y_shift = radius * np.sin(np.radians(angle))
    
    
#     candidate_poses = [
#                 [0, y_shift, z_shift, np.radians(angle), 0 ,0 ], # up-front
#                 [x_shift, 0, z_shift, 0, np.radians(angle), 0], # right-front
#                 [0, -y_shift, z_shift, np.radians(-angle), 0 ,0 ], # down-front,
#                 [x_shift, 0, z_shift, 0, np.radians(-angle), 0], # left-front
#                 [0, y_shift, -z_shift, np.radians(angle), 0 ,0 ], # up-back
#                 [x_shift, 0, -z_shift, 0, np.radians(angle), 0], # right-back
#                 [0, -y_shift, -z_shift, np.radians(-angle), 0 ,0 ], # down,-back
#                 [x_shift, 0, -z_shift, 0, np.radians(-angle), 0], # left-back
#             ]

    candidate_poses = [
                [0, y_shift, z_shift, 0, 0 ,0 ], # up-front
                [x_shift, 0, z_shift, 0, np.radians(angle), 0], # right-front
                [0, -y_shift, z_shift, 0, 0 ,0 ], # down-front,
                [x_shift, 0, z_shift, 0, np.radians(-angle), 0], # left-front
                [0, y_shift, -z_shift, 0, 0 ,0 ], # up-back
                [x_shift, 0, -z_shift, 0, np.radians(angle), 0], # right-back
                [0, -y_shift, -z_shift, 0, 0 ,0 ], # down,-back
                [x_shift, 0, -z_shift, 0, np.radians(-angle), 0], # left-back
            ]
    
    
    proj_img_list = []
    proj_color_list = []
    num_pts_list = []
    pose_list = []
    pcd_list = []
    
    for pose in candidate_poses:
        temp_pcd = copy.copy(pcd)
        
        temp_pcd.translate([-pose[0], -pose[1], -pose[2]])
        R = temp_pcd.get_rotation_matrix_from_axis_angle(np.array(pose[3:6]).T)
        temp_pcd.rotate(R)
        
        
        pcd_list.append(copy.deepcopy(temp_pcd))
        
        ## Finding hidden points from this location
        ## Since we have alreay rotateted and translated the point clould, the view point is origin
        hid_ids = nbv_planner.remove_hidden_point_from_loc(temp_pcd)
        
        ## Keep only visible points
        visible_predicted_points = list(set(diff_ids).intersection(hid_ids))
        temp_pcd = temp_pcd.select_by_index(visible_predicted_points)
        
#         pcd_list.append(copy.deepcopy(temp_pcd))
        
        data = np.asarray(temp_pcd.points)
        proj_pts = data.copy() @ intrinsic.T

        depth = data[:,2][:,None]
        proj_pts = proj_pts/depth
        proj_img_pts = (proj_pts.round()).astype(int)[:,:2]
        proj_img_pts = proj_img_pts[(proj_img_pts[:,0] >= 0) & (proj_img_pts[:,0] < im_width) & (proj_img_pts[:,1] >= 0) & (proj_img_pts[:,1] < im_height)]
        proj_img_pts, z_index = np.unique(proj_img_pts[:,[0,1]], axis=0, return_index=True)


#         pose_list.append([rob_x, rob_y, yaw, ma])
        proj_img_list.append(proj_img_pts.copy())
        proj_color_list.append((255*(depth[z_index])/10).astype(np.uint8))
        num_pts_list.append(len(proj_img_pts))

        del temp_pcd
        
        
#     return pcd_list, proj_img_list
    return proj_img_list, proj_color_list, num_pts_list, candidate_poses, pcd_list


def NBV2D(pred_pcd, diff_ids):
    im_height, im_width = 480, 480
    intrinsic = np.array([[im_width//2, 0, im_width//2],
                          [0, im_height//2, im_height//2],
                          [0, 0, 1]])

    radius = 0.05
#     yaw_list = [-60, -45, -30, 0, 30, 45, 60]
#     move_angle = [-60, -45, -30, 0, 30, 45, 60]

    yaw_list = [-60, -45, -30, 0, 30, 45, 60]
    move_angle = [-120, -90, -60, -45, -30, 0, 30, 45, 60, 90, 120]

#     height_list = [0, 0.2]

    proj_img_list = []
    proj_color_list = []
    num_pts_list = []
    pose_list = []

    pcd_list = []
    for ma in move_angle:
        for yaw in yaw_list:
            rob_x = radius * np.sin(ma*np.pi/180)
            rob_y = radius * np.cos(ma*np.pi/180)

            pcd = copy.copy(pred_pcd)

    #         sub_indices = np.argwhere(np.asarray(pcd.points)[:,2] > 1)
    #         pcd = pcd.select_by_index(sub_indices)

            # Left rotation is +ve
            R = pcd.get_rotation_matrix_from_axis_angle(np.array([0, np.radians(yaw), 0]).T)
            # pcd.translate([1-rob_x, 0, -rob_y])
            pcd.translate([-rob_x, 0, -rob_y])
            pcd.rotate(R)
            
            
            all_data = np.asarray(pcd.points)
            
            pcd_list.append(copy.deepcopy(pcd))
            ## Finding hidden points from this location
            ## Since we have alreay rotateted and translated the point clould, the view point is origin
            hid_ids = nbv_planner.remove_hidden_point_from_loc(pcd)
            
            """
            ## Keep only visible points
            visible_predicted_points = list(set(diff_ids).intersection(hid_ids))
            pcd = pcd.select_by_index(visible_predicted_points)
            """
            ## Selcting only new points and later will check if the drone collides with the object
            pcd = pcd.select_by_index(diff_ids)
            
#             pcd_list.append(copy.deepcopy(pcd))

            data = np.asarray(pcd.points)
            proj_pts = data.copy() @ intrinsic.T

            depth = data[:,2][:,None]
            proj_pts = proj_pts/depth
            proj_img_pts = (proj_pts.round()).astype(int)[:,:2]
            proj_img_pts = proj_img_pts[(proj_img_pts[:,0] >= 0) & (proj_img_pts[:,0] < im_width) & (proj_img_pts[:,1] >= 0) & (proj_img_pts[:,1] < im_height)]
            proj_img_pts, z_index = np.unique(proj_img_pts[:,[0,1]], axis=0, return_index=True)


            pose_list.append([rob_x, 0, rob_y, 0, np.radians(yaw), 0])
            proj_img_list.append(proj_img_pts.copy())
            proj_color_list.append((255*(depth[z_index])/10).astype(np.uint8))
            
            if np.sign(all_data[:,2].min()) != np.sign(all_data[:,2].max()): ## if center is between min and max depth
                num_pts_list.append(-2)
            else:
                num_pts_list.append(len(proj_img_pts))

            del pcd
    return proj_img_list, proj_color_list, num_pts_list, pose_list, pcd_list
