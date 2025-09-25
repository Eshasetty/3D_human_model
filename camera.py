"""
Camera module for 3D human body visualization.
Contains the HumanBodyCamera class for perspective projection.
"""

import numpy as np


def _normalize(vector):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class HumanBodyCamera:
    """Camera class for 3D perspective projection."""
    
    def __init__(self, image_width=800, image_height=600):
        # Camera intrinsic parameters
        self.alpha = 500    # Focal length in x direction
        self.beta = 500     # Focal length in y direction (usually same as alpha)
        self.x0 = image_width / 2  # Principal point x
        self.y0 = image_height / 2  # Principal point y
        self.image_width = image_width
        self.image_height = image_height
        
        # Build intrinsic matrix
        self.K = self._build_intrinsic_matrix()
    
    def _build_intrinsic_matrix(self):
        """Build the camera intrinsic matrix K."""
        return np.array([[self.alpha, 0, self.x0],
                        [0, self.beta, self.y0],
                        [0, 0, 1]])
    
    def _calculate_extrinsics(self, camera_position, camera_direction):
        """Calculate rotation matrix R and translation vector t for camera pose."""
        # Normalize direction
        forward = _normalize(camera_direction)
        world_up = np.array([0, 0, 1])  # Z-up world
        
        # Handle case where forward is parallel to world up
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([1, 0, 0])
            
        right = _normalize(np.cross(world_up, forward))
        up = _normalize(np.cross(forward, right))
        
        # Camera coordinate frame: x=right, y=up, z=forward (into scene is negative z)
        R = np.vstack([right, up, -forward])
        t = -R @ np.array(camera_position)
        return R, t
    
    def project_points(self, world_points, camera_position, camera_direction):
        """
        Project 3D world points to 2D image coordinates.
        
        Args:
            world_points: Array of 3D points (N x 3)
            camera_position: 3D position of camera
            camera_direction: 3D direction vector camera is looking
            
        Returns:
            Array of 2D projected points (N x 2)
        """
        # Calculate extrinsic parameters
        R, t = self._calculate_extrinsics(camera_position, camera_direction)
        
        # Build projection matrix P = K[R|t]
        P = self.K @ np.hstack([R, t.reshape(-1, 1)])
        
        # Convert to homogeneous coordinates
        if world_points.shape[1] == 3:
            world_points_hom = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
        else:
            world_points_hom = world_points
        
        # Project to image coordinates
        image_points_hom = (P @ world_points_hom.T).T
        
        # Convert from homogeneous to 2D coordinates
        image_points = image_points_hom[:, :2] / image_points_hom[:, 2:3]
        
        return image_points
