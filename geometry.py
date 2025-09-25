"""
Geometry module for 3D human body visualization.
Contains functions for creating 3D shapes and geometric primitives.
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_sphere(center, radius, resolution=20):
    """
    Create a sphere mesh.
    
    Args:
        center: (x, y, z) center coordinates
        radius: sphere radius
        resolution: number of points along each axis
        
    Returns:
        X, Y, Z coordinate arrays for plotting
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    return x, y, z


def create_cylinder(center, radius, length, axis='z', resolution=20):
    """
    Create a cylinder mesh.
    
    Args:
        center: (x, y, z) center coordinates
        radius: cylinder radius
        length: cylinder length
        axis: 'x', 'y', or 'z' for cylinder orientation
        resolution: number of points around circumference
        
    Returns:
        X, Y, Z coordinate arrays for plotting
    """
    theta = np.linspace(0, 2 * np.pi, resolution)
    h = np.linspace(-length/2, length/2, 2)
    
    if axis == 'x':
        x = np.outer(np.ones(len(theta)), h) + center[0]
        y = radius * np.outer(np.cos(theta), np.ones(len(h))) + center[1]
        z = radius * np.outer(np.sin(theta), np.ones(len(h))) + center[2]
    elif axis == 'y':
        x = radius * np.outer(np.cos(theta), np.ones(len(h))) + center[0]
        y = np.outer(np.ones(len(theta)), h) + center[1]
        z = radius * np.outer(np.sin(theta), np.ones(len(h))) + center[2]
    else:  # axis == 'z'
        x = radius * np.outer(np.cos(theta), np.ones(len(h))) + center[0]
        y = radius * np.outer(np.sin(theta), np.ones(len(h))) + center[1]
        z = np.outer(np.ones(len(theta)), h) + center[2]
    
    return x, y, z


def create_box(center, width, height, depth):
    """
    Create a box (rectangular prism) as a collection of faces.
    
    Args:
        center: (x, y, z) center coordinates
        width: box width (x dimension)
        height: box height (y dimension)
        depth: box depth (z dimension)
        
    Returns:
        List of face polygons for Poly3DCollection
    """
    x, y, z = center
    w, h, d = width/2, height/2, depth/2
    
    # Define the 8 vertices of the box
    vertices = [
        [x-w, y-h, z-d], [x+w, y-h, z-d], [x+w, y+h, z-d], [x-w, y+h, z-d],  # bottom
        [x-w, y-h, z+d], [x+w, y-h, z+d], [x+w, y+h, z+d], [x-w, y+h, z+d]   # top
    ]
    
    # Define the 6 faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    return faces


def add_ground_plane(ax, size=2.5, grid_step=0.25, z=0.0, color='#2C2F3A', alpha=0.35, 
                    grid_color='#3E4454', grid_alpha=0.25):
    """
    Add a ground plane to the 3D plot.
    
    Args:
        ax: matplotlib 3D axes
        size: size of the ground plane
        grid_step: spacing between grid lines
        z: z-coordinate of the ground plane
        color: color of the ground plane
        alpha: transparency of the ground plane
        grid_color: color of the grid lines
        grid_alpha: transparency of the grid lines
    """
    # Create grid
    x = np.linspace(-size, size, int(2*size/grid_step) + 1)
    y = np.linspace(-size, size, int(2*size/grid_step) + 1)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z)
    
    # Plot the ground plane
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)
    
    # Add grid lines
    for i in range(0, len(x), 2):
        ax.plot(x[i], y, z, color=grid_color, alpha=grid_alpha, linewidth=0.5)
    for i in range(0, len(y), 2):
        ax.plot(x, y[i], z, color=grid_color, alpha=grid_alpha, linewidth=0.5)


def surface_to_quads(X, Y, Z):
    """
    Convert surface mesh to quads for depth sorting.
    
    Args:
        X, Y, Z: coordinate arrays
        
    Returns:
        List of (depth, quad_points) tuples
    """
    quads = []
    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            # Get the four corners of the quad
            x1, y1, z1 = X[i, j], Y[i, j], Z[i, j]
            x2, y2, z2 = X[i+1, j], Y[i+1, j], Z[i+1, j]
            x3, y3, z3 = X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]
            x4, y4, z4 = X[i, j+1], Y[i, j+1], Z[i, j+1]
            
            # Calculate average depth
            avg_depth = (z1 + z2 + z3 + z4) / 4
            quad_points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            quads.append((avg_depth, quad_points))
    
    return quads
