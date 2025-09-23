import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def create_sphere(center, radius, resolution=20):
    """Create a sphere using spherical coordinates"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def create_cylinder(center, radius, height, axis='z', resolution=20):
    """Create a cylinder along specified axis"""
    theta = np.linspace(0, 2*np.pi, resolution)
    
    if axis == 'z':
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z_bottom = np.full_like(x, center[2] - height/2)
        z_top = np.full_like(x, center[2] + height/2)
        
        # Create cylinder surface
        X = np.array([x, x])
        Y = np.array([y, y])
        Z = np.array([z_bottom, z_top])
        
    elif axis == 'x':
        y = center[1] + radius * np.cos(theta)
        z = center[2] + radius * np.sin(theta)
        x_bottom = np.full_like(y, center[0] - height/2)
        x_top = np.full_like(y, center[0] + height/2)
        
        X = np.array([x_bottom, x_top])
        Y = np.array([y, y])
        Z = np.array([z, z])
        
    elif axis == 'y':
        x = center[0] + radius * np.cos(theta)
        z = center[2] + radius * np.sin(theta)
        y_bottom = np.full_like(x, center[1] - height/2)
        y_top = np.full_like(x, center[1] + height/2)
        
        X = np.array([x, x])
        Y = np.array([y_bottom, y_top])
        Z = np.array([z, z])
    
    return X, Y, Z

def create_box(center, width, height, depth):
    """Create a rectangular box"""
    x, y, z = center
    w, h, d = width/2, height/2, depth/2
    
    # Define the vertices of the box
    vertices = np.array([
        [x-w, y-h, z-d], [x+w, y-h, z-d], [x+w, y+h, z-d], [x-w, y+h, z-d],  # bottom
        [x-w, y-h, z+d], [x+w, y-h, z+d], [x+w, y+h, z+d], [x-w, y+h, z+d]   # top
    ])
    
    # Define the faces using vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    return faces

def plot_geometric_human():
    """Create and plot a geometric human model"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set the background color to match the purple gradient
    fig.patch.set_facecolor('#6B5B95')
    ax.set_facecolor('#4A4A4A')
    
    # Head (sphere)
    head_center = (0, 0, 2.5)
    head_radius = 0.3
    head_x, head_y, head_z = create_sphere(head_center, head_radius)
    ax.plot_surface(head_x, head_y, head_z, color='#D2B48C', alpha=0.9)
    
    # Torso (rectangular box)
    torso_center = (0, 0, 1.5)
    torso_faces = create_box(torso_center, 0.8, 0.4, 1.2)
    torso_collection = Poly3DCollection(torso_faces, facecolors='#4A90E2', alpha=0.9, edgecolors='black')
    ax.add_collection3d(torso_collection)
    
    # Left arm (cylinder)
    left_arm_center = (-0.7, 0, 1.8)
    left_arm_x, left_arm_y, left_arm_z = create_cylinder(left_arm_center, 0.1, 0.8, axis='x')
    ax.plot_surface(left_arm_x, left_arm_y, left_arm_z, color='#4A90E2', alpha=0.9)
    
    # Right arm (cylinder)
    right_arm_center = (0.7, 0, 1.8)
    right_arm_x, right_arm_y, right_arm_z = create_cylinder(right_arm_center, 0.1, 0.8, axis='x')
    ax.plot_surface(right_arm_x, right_arm_y, right_arm_z, color='#4A90E2', alpha=0.9)
    
    # Left leg (cylinder)
    left_leg_center = (-0.2, 0, 0.4)
    left_leg_x, left_leg_y, left_leg_z = create_cylinder(left_leg_center, 0.12, 0.8, axis='z')
    ax.plot_surface(left_leg_x, left_leg_y, left_leg_z, color='#4A90E2', alpha=0.9)
    
    # Right leg (cylinder)
    right_leg_center = (0.2, 0, 0.4)
    right_leg_x, right_leg_y, right_leg_z = create_cylinder(right_leg_center, 0.12, 0.8, axis='z')
    ax.plot_surface(right_leg_x, right_leg_y, right_leg_z, color='#4A90E2', alpha=0.9)
    
    # Hands (small spheres)
    hand_radius = 0.08
    
    # Left hand
    left_hand_center = (-1.1, 0, 1.8)
    left_hand_x, left_hand_y, left_hand_z = create_sphere(left_hand_center, hand_radius, resolution=10)
    ax.plot_surface(left_hand_x, left_hand_y, left_hand_z, color='#D2B48C', alpha=0.9)
    
    # Right hand
    right_hand_center = (1.1, 0, 1.8)
    right_hand_x, right_hand_y, right_hand_z = create_sphere(right_hand_center, hand_radius, resolution=10)
    ax.plot_surface(right_hand_x, right_hand_y, right_hand_z, color='#D2B48C', alpha=0.9)
    
    # Feet (small boxes)
    foot_size = (0.3, 0.15, 0.1)
    
    # Left foot
    left_foot_center = (-0.2, 0.1, 0)
    left_foot_faces = create_box(left_foot_center, *foot_size)
    left_foot_collection = Poly3DCollection(left_foot_faces, facecolors='#2C3E50', alpha=0.9, edgecolors='black')
    ax.add_collection3d(left_foot_collection)
    
    # Right foot
    right_foot_center = (0.2, 0.1, 0)
    right_foot_faces = create_box(right_foot_center, *foot_size)
    right_foot_collection = Poly3DCollection(right_foot_faces, facecolors='#2C3E50', alpha=0.9, edgecolors='black')
    ax.add_collection3d(right_foot_collection)
    
    # Set the aspect ratio and limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 3])
    
    # Set title on the figure (keep visible when axes are hidden)
    fig.suptitle('Geometric Human Model', fontsize=16, color='white')
    
    # Remove axes for a cleaner look
    ax.set_axis_off()
    
    # Set viewing angle
    ax.view_init(elev=10, azim=45)
    
    plt.tight_layout()
    plt.show()

# Alternative version using PyOpenGL for more advanced rendering
def create_opengl_version():
    """
    Alternative implementation using PyOpenGL (requires: pip install PyOpenGL PyOpenGL_accelerate)
    Uncomment and run this if you want hardware-accelerated 3D rendering
    """
    print("OpenGL version requires additional packages:")
    print("pip install PyOpenGL PyOpenGL_accelerate pygame")
    print("This version would provide better performance and more realistic lighting.")

if __name__ == "__main__":
    # Create and display the geometric human model
    plot_geometric_human()
    
    # Print information about the alternative version
    print("\nGeometric Human Model created!")
    print("- Head: Sphere (skin color)")
    print("- Torso: Rectangular box (blue)")
    print("- Arms: Cylinders (blue)")
    print("- Legs: Cylinders (blue)")
    print("- Hands: Small spheres (skin color)")
    print("- Feet: Small rectangular boxes (dark color)")
    print("\nYou can modify colors, sizes, and positions by adjusting the parameters in the code.")