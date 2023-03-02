import numpy as np
import open3d as o3d
import copy

class Pose2Image():
    def __init__(self, height=480, width=480):
        self.intrinsic = np.array([[width//2, 0, width//2],
                              [0, height//2, height//2],
                              [0, 0, 1]])
        
        self.height = height
        self.width = width
        
        self.source_pcd = o3d.geometry.PointCloud()
        self.source_pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
        
    
    def get_image(self, point_cloud, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, diff_ids=[]):
        pcd = copy.deepcopy(point_cloud)
        
        R = pcd.get_rotation_matrix_from_axis_angle(np.array([np.radians(-roll), np.radians(-yaw), np.radians(-pitch)]).T)
        pcd = pcd.translate([-x, -y, -z], relative=False)
        pcd = pcd.rotate(R, center=(0,0,0))
        
        
        all_data = np.asarray(pcd.points)

        pcd4demo = copy.deepcopy(pcd) #+ self.source_pcd
        ## Finding hidden points from this location
        ## Since we have alreay rotateted and translated the point clould, the view point is origin
        hid_ids = self.remove_hidden_point_from_loc(pcd)
        
        ## Keep only visible points
        visible_predicted_points = list(set(diff_ids).intersection(hid_ids))
        pcd = pcd.select_by_index(visible_predicted_points)
        
        
        data = np.asarray(pcd.points)
        proj_pts = data.copy() @ self.intrinsic.T

        depth = data[:,2][:,None]
        proj_pts = proj_pts/depth
        proj_img_pts = (proj_pts.round()).astype(int)[:,:2]
        proj_img_pts = proj_img_pts[(proj_img_pts[:,0] >= 0) & (proj_img_pts[:,0] < self.width) & (proj_img_pts[:,1] >= 0) & (proj_img_pts[:,1] < self.height)]
        proj_img_pts, z_index = np.unique(proj_img_pts[:,[0,1]], axis=0, return_index=True)
        
        proj_img_colors = (255*(depth[z_index])/10).astype(np.uint8)
        
        del pcd
        return proj_img_pts, proj_img_colors, pcd4demo, hid_ids# pcd, hid_ids
    
    def remove_points_from_1(self, pcd1, pcd2, thresh=1e-5):
        pts1 = np.asarray(pcd1.points)
        pts2 = np.asarray(pcd2.points)

        pts_set = []
        for p in pts1:
            if np.linalg.norm(pts2 - p, axis=1).min() > thresh:
                pts_set.append(p)

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.array(pts_set))

        return new_pcd
    
    def get_nonoverlapping_points(self, pcd1, pcd2, thresh):
        """
        diff_ids = get_nonoverlapping_points(pred_pcd, pcd_partial)
        plot_3d_jet(pred_pcd.select_by_index(diff_ids))
        """
        pts1 = np.asarray(pcd1.points)
        pts2 = np.asarray(pcd2.points)

        pts_ids = []
        for i in range(len(pts1)):
            if np.linalg.norm(pts2 - pts1[i], axis=1).min() > thresh:
                pts_ids.append(i)

        return pts_ids


    def remove_hidden_point_from_loc(self, pcd):
        """
        hid_ids = remove_hidden_point_from_loc(pred_pcd, x=x_coord, y=0, z=z_coord, rx=0, ry=z_rot, rz=0)
        plot_3d_jet(pred_pcd.select_by_index(hid_ids))
        """
        pcd_temp = copy.deepcopy(pcd)
#         pcd_temp = orient(pcd_temp, x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)

        diameter = np.linalg.norm(np.asarray(pcd_temp.get_max_bound()) - np.asarray(pcd_temp.get_min_bound()))
        camera = [0, 0, 0] # camera is assumed to be at 0,0,0, because we moved the point cloud instead
        radius = diameter * 100
        _, pt_map = pcd_temp.hidden_point_removal(camera, radius)

        return pt_map
    
    def depth_image_for_pcd(self, pcd):
        ## Moving the camera
        pcd_temp = copy.deepcopy(pcd)
        
        """
        pcd_temp = pcd_temp.translate((-x,-y,-z), relative=False)
        
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(-rx), np.radians(-ry), np.radians(-rz)])
        pcd_temp = pcd_temp.rotate(R, center=(0,0,0))
        """
        
        # return pcd_temp
        ## Occlusion/Hidden Point Removal
        diameter = np.linalg.norm(np.asarray(pcd_temp.get_max_bound()) - np.asarray(pcd_temp.get_min_bound()))

        camera = [0, 0, 0] # camera is assumed to be at 0,0,0, because we moved the point cloud instead
        radius = diameter * 100
        _, pt_map = pcd_temp.hidden_point_removal(camera, radius)
        pcd_temp = pcd_temp.select_by_index(pt_map)

        ## Depth Image generation
        pcd_l = o3d.t.geometry.PointCloud.from_legacy(pcd_temp)

        depth_reproj = pcd_l.project_to_depth_image(width=self.width, height=self.height, 
                                 intrinsics=o3d.core.Tensor(self.intrinsic),
                                 # extrinsics=self.get_extrinsic(x=0, y=0, z=0, ry=0),
                                 depth_scale=1.,
                                 depth_max=10.)

        depth_img = np.asarray(depth_reproj.to_legacy())
        
        del pcd_temp
        
        return depth_img #, pcd_temp
    

#     visible_predicted_points = list(set(diff_ids).intersection(hid_ids))
#     plot_3d_jet(pred_pcd.select_by_index(visible_predicted_points))