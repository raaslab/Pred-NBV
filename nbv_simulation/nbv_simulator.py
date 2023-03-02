import numpy as np
import open3d as o3d
import copy
import plotly.graph_objects as go

class NBV_Simulator():
    def __init__(self, pcd_file, voxel_size = 0.05):
        pcd_np = np.load(pcd_file)
        
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pcd_np)
        self.voxel_size = voxel_size
        
    def get_intrinsic_matrix_tensor(self, width=480, height=480, fov=90, as_tensor=True):
        """Function to calculate the intrinsic matrix for the robot-mounted (ans the center, left, and right) cameras

        Parameters
        ----------
            None

        Returns
        -------
            open3d intrindic matrix object: A 3x3 array  intrinsic matrix object
        """
        # Helper function to convert degrees to radian
        def to_rad(th):
            return th*math.pi / 180

        # Convert the FoVs to focal lengths
        focal_length_x = 0.5 * width * np.tan(np.radians(fov/2))
        focal_length_y = 0.5 * height * np.tan(np.radians(fov/2))

        # Collect the parameters for the intrinsic matrix (focal length for x-axis, focal length for y-axis, center for x-axis, center for y-axis)
        fx, fy, cx, cy = (focal_length_x, focal_length_y, width/2, height/2)

        # Get the inmtrinsic matrix assuming Pinhole Camera using Open3D
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Return the intrinsic matrix
        if as_tensor:
            return o3d.core.Tensor(intrinsic.intrinsic_matrix)
        else:
            return intrinsic
        
    def get_extrinsic(self, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0):
        extrinsic = np.eye(4)
        extrinsic[:3,  3] = (x, y, z)
        extrinsic[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(rx), np.radians(ry), np.radians(rz)])
        return extrinsic
    
    def depth_image_from_pose(self, x=0, y=0, z=0, rx=0, ry=0, rz=0):
        ## Moving the camera
        pcd_temp = copy.deepcopy(self.pcd)
        
        pcd_temp = pcd_temp.translate((-x,-y,-z), relative=False)
        
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(-rx), np.radians(-ry), np.radians(-rz)])
        pcd_temp = pcd_temp.rotate(R, center=(0,0,0))
        
        
        # return pcd_temp
        ## Occlusion/Hidden Point Removal
        diameter = np.linalg.norm(np.asarray(pcd_temp.get_max_bound()) - np.asarray(pcd_temp.get_min_bound()))

        camera = [0, 0, 0] # camera is assumed to be at 0,0,0, because we moved the point cloud instead
        radius = diameter * 100
        _, pt_map = pcd_temp.hidden_point_removal(camera, radius)
        pcd_temp = pcd_temp.select_by_index(pt_map)

        ## Depth Image generation
        pcd_l = o3d.t.geometry.PointCloud.from_legacy(pcd_temp)

        depth_reproj = pcd_l.project_to_depth_image(width=480, height=480, 
                                 intrinsics=self.get_intrinsic_matrix_tensor(width=480, height=480, fov=90),
                                 extrinsics=self.get_extrinsic(x=0, y=0, z=0, ry=0),
                                 depth_scale=1.,
                                 depth_max=10.)

        depth_img = np.asarray(depth_reproj.to_legacy())

        return depth_img #, pcd_temp
    
    def preprocess_point_cloud(self, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = self.pcd.voxel_down_sample(voxel_size)

        radius_normal = self.voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def preprocess_point_cloud_full_res(self, voxel_size):
        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd, pcd_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = self.voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = self.voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return result

def NBV2D(pred_pcd, diff_ids):
    im_height, im_width = 480, 480
    intrinsic = np.array([[im_width//2, 0, im_width//2],
                          [0, im_height//2, im_height//2],
                          [0, 0, 1]])

    radius = 0.2
    yaw_list = [-60, -45, -30, 0, 30, 45, 60]
    move_angle = [-60, -45, -30, 0, 30, 45, 60]
    height_list = [0, 0.2]

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

            ## Finding hidden points from this location
            ## Since we have alreay rotateted and translated the point clould, the view point is origin
            hid_ids = nbv_planner.remove_hidden_point_from_loc(pcd)

            ## Keep only visible points
            visible_predicted_points = list(set(diff_ids).intersection(hid_ids))
            pcd = pcd.select_by_index(visible_predicted_points)
    #             pcd = pcd.select_by_index(hid_ids)

            pcd_list.append(copy.deepcopy(pcd))

            data = np.asarray(pcd.points)
            proj_pts = data.copy() @ intrinsic.T

            depth = data[:,2][:,None]
            proj_pts = proj_pts/depth
            proj_img_pts = (proj_pts.round()).astype(int)[:,:2]
            proj_img_pts = proj_img_pts[(proj_img_pts[:,0] >= 0) & (proj_img_pts[:,0] < im_width) & (proj_img_pts[:,1] >= 0) & (proj_img_pts[:,1] < im_height)]
            proj_img_pts, z_index = np.unique(proj_img_pts[:,[0,1]], axis=0, return_index=True)


            pose_list.append([rob_x, rob_y, yaw, ma])
            proj_img_list.append(proj_img_pts.copy())
            proj_color_list.append((255*(depth[z_index])/10).astype(np.uint8))
            num_pts_list.append(len(proj_img_pts))

            del pcd
    return proj_img_list, proj_color_list, num_pts_list, pose_list

################################################################################
########################### Plotting ###########################################
################################################################################
def plot_3d(points_raw, colors):
    if isinstance(points_raw, np.ndarray):
        points = points_raw
    else:
        points = np.asarray(points_raw.points)
    
    trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=list(map(lambda e: 'rgb('+', '.join(e.astype(str))+')', colors[:,::-1])),
            ),
        )
    data = [trace]
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def plot_3d_jet(points_raw):
    if isinstance(points_raw, np.ndarray):
        points = points_raw
    else:
        points = np.asarray(points_raw.points)
        
    trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=np.linalg.norm(points, axis=1),
                colorscale='hsv'
            ),
            # color='jet',
        )
    data = [trace]
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    fig.show()

################################################################################
########################### PC Manipulation ####################################
################################################################################
def reorient(pcd, x=0,y=0, z=0, rx=0, ry=0, rz=0):
    pcd_temp = copy.deepcopy(pcd)
            
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(rx), np.radians(ry), np.radians(rz)])
    pcd_temp = pcd_temp.rotate(R, center=(0,0,0))

    pcd_temp = pcd_temp.translate((x,y,z), relative=True)
    
    return pcd_temp

def orient(pcd, x=0,y=0, z=0, rx=0, ry=0, rz=0):
    pcd_temp = copy.deepcopy(pcd)
    
    pcd_temp = pcd_temp.translate((-x,-y,-z), relative=False)
    
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(-rx), np.radians(-ry), np.radians(-rz)])
    pcd_temp = pcd_temp.rotate(R, center=(0,0,0))
    
    return pcd_temp


################################################################################
################################ ICP ###########################################
################################################################################
def preprocess_point_cloud(pcd, voxel_size, verbose=False):
    if verbose:
        print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    if verbose:
        print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    if verbose:
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, verbose=False):
    distance_threshold = voxel_size * 1.5
    if verbose:
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac, verbose=False):
    distance_threshold = voxel_size * 0.4
    if verbose:
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, #result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result
    
def apply_icp(source, target, voxel_size=0.05, verbose=False):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size=voxel_size, verbose=verbose)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=voxel_size, verbose=verbose)
    
    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    
    source.transform(result_ransac.transformation)
    
#     result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
#                                  voxel_size, None) #result_ransac)
    
#     source.transform(result_icp.transformation)


    return source