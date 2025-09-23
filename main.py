import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def get_default_params():
    """Return default per-part parameters for the geometric human model."""
    return {
        'scale': 1.0,
        'ground': {'size': 2.5, 'grid_step': 0.25, 'z': 0.0},
        'head': {'center': (0.0, 0.0, 1.65), 'radius': 0.12},
        'torso': {'center': (0.0, 0.0, 1.1), 'width': 0.4, 'height': 0.25, 'depth': 0.6},
        'left_upper_arm': {'center': (-0.35, 0.0, 1.25), 'radius': 0.06, 'length': 0.35, 'axis': 'x'},
        'left_forearm': {'center': (-0.65, 0.0, 1.25), 'radius': 0.05, 'length': 0.35, 'axis': 'x'},
        'right_upper_arm': {'center': (0.35, 0.0, 1.25), 'radius': 0.06, 'length': 0.35, 'axis': 'x'},
        'right_forearm': {'center': (0.65, 0.0, 1.25), 'radius': 0.05, 'length': 0.35, 'axis': 'x'},
        'left_thigh': {'center': (-0.12, 0.0, 0.625), 'radius': 0.07, 'length': 0.35, 'axis': 'z'},
        'left_shin': {'center': (-0.12, 0.0, 0.275), 'radius': 0.06, 'length': 0.35, 'axis': 'z'},
        'right_thigh': {'center': (0.12, 0.0, 0.625), 'radius': 0.07, 'length': 0.35, 'axis': 'z'},
        'right_shin': {'center': (0.12, 0.0, 0.275), 'radius': 0.06, 'length': 0.35, 'axis': 'z'},
        'hands': {
            'radius': 0.05,
            'left_center': (-0.72, 0.0, 1.25),
            'right_center': (0.72, 0.0, 1.25)
        },
        'feet': {
            'size': (0.18, 0.10, 0.06),
            'left_center': (-0.12, 0.08, 0.03),
            'right_center': (0.12, 0.08, 0.03)
        },
        'limits': {'x': 1.5, 'y': 1.5, 'z': 2.0}
    }

def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively update mapping `base` with `overrides` and return it."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

def add_ground_plane(ax, size=2.5, grid_step=0.25, z=0.0):
    """Add a flat ground plane with a subtle grid at height z."""
    half = size
    plane_vertices = np.array([
        [-half, -half, z],
        [ half, -half, z],
        [ half,  half, z],
        [-half,  half, z]
    ])
    plane_face = [plane_vertices[0], plane_vertices[1], plane_vertices[2], plane_vertices[3]]

    plane = Poly3DCollection([plane_face], facecolors='#2C2F3A', alpha=0.9, edgecolors='none')
    ax.add_collection3d(plane)

    # draw grid lines
    grid_color = '#3E4454'
    al = 0.35
    ticks = np.arange(-half, half + 1e-6, grid_step)
    for t in ticks:
        ax.plot([-half, half], [t, t], [z, z], color=grid_color, alpha=al, linewidth=0.8)
        ax.plot([t, t], [-half, half], [z, z], color=grid_color, alpha=al, linewidth=0.8)

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

def plot_geometric_human(params: dict | None = None):
    """Create and plot a geometric human model using per-part parameters.

    params: optional dict overriding defaults from `get_default_params()`.
    Supports a global 'scale' applied to all linear dimensions.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set the background color to match the purple gradient
    fig.patch.set_facecolor('#6B5B95')
    ax.set_facecolor('#4A4A4A')
    
    # Merge params with defaults and compute scale
    p = deep_update(get_default_params(), params or {})
    s = float(p['scale'])

    # Ground plane
    add_ground_plane(
        ax,
        size=p['ground']['size'] * s,
        grid_step=p['ground']['grid_step'] * s,
        z=p['ground']['z']
    )

    # Head (sphere) — size from radius formula r = r0 * s
    head_center = tuple(np.array(p['head']['center']) * np.array([s, s, s]))
    head_radius = p['head']['radius'] * s
    head_x, head_y, head_z = create_sphere(head_center, head_radius)
    ax.plot_surface(head_x, head_y, head_z, color='#D2B48C', alpha=0.9)
    
    # Torso (rectangular box) — dimensions scale linearly with s
    torso_center = tuple(np.array(p['torso']['center']) * np.array([s, s, s]))
    torso_faces = create_box(
        torso_center,
        p['torso']['width'] * s,
        p['torso']['height'] * s,
        p['torso']['depth'] * s
    )
    torso_collection = Poly3DCollection(torso_faces, facecolors='#4A90E2', alpha=0.9, edgecolors='black')
    ax.add_collection3d(torso_collection)
    
    # Arms: split into upper arm (larger) and forearm (smaller)
    # Left arm
    lua_c = p['left_upper_arm']
    left_upper_arm_center = tuple(np.array(lua_c['center']) * np.array([s, s, s]))
    lua_x, lua_y, lua_z = create_cylinder(left_upper_arm_center, lua_c['radius'] * s, lua_c['length'] * s, axis=lua_c['axis'])
    ax.plot_surface(lua_x, lua_y, lua_z, color='#4A90E2', alpha=0.9)
    lfa_c = p['left_forearm']
    left_forearm_center = tuple(np.array(lfa_c['center']) * np.array([s, s, s]))
    lfa_x, lfa_y, lfa_z = create_cylinder(left_forearm_center, lfa_c['radius'] * s, lfa_c['length'] * s, axis=lfa_c['axis'])
    ax.plot_surface(lfa_x, lfa_y, lfa_z, color='#4A90E2', alpha=0.9)

    # Right arm
    rua_c = p['right_upper_arm']
    right_upper_arm_center = tuple(np.array(rua_c['center']) * np.array([s, s, s]))
    rua_x, rua_y, rua_z = create_cylinder(right_upper_arm_center, rua_c['radius'] * s, rua_c['length'] * s, axis=rua_c['axis'])
    ax.plot_surface(rua_x, rua_y, rua_z, color='#4A90E2', alpha=0.9)
    rfa_c = p['right_forearm']
    right_forearm_center = tuple(np.array(rfa_c['center']) * np.array([s, s, s]))
    rfa_x, rfa_y, rfa_z = create_cylinder(right_forearm_center, rfa_c['radius'] * s, rfa_c['length'] * s, axis=rfa_c['axis'])
    ax.plot_surface(rfa_x, rfa_y, rfa_z, color='#4A90E2', alpha=0.9)
    
    # Legs: split into thigh (larger) and shin (smaller)
    # Left leg
    lth_c = p['left_thigh']
    left_thigh_center = tuple(np.array(lth_c['center']) * np.array([s, s, s]))
    lth_x, lth_y, lth_z = create_cylinder(left_thigh_center, lth_c['radius'] * s, lth_c['length'] * s, axis=lth_c['axis'])
    ax.plot_surface(lth_x, lth_y, lth_z, color='#4A90E2', alpha=0.9)
    lsh_c = p['left_shin']
    left_shin_center = tuple(np.array(lsh_c['center']) * np.array([s, s, s]))
    lsh_x, lsh_y, lsh_z = create_cylinder(left_shin_center, lsh_c['radius'] * s, lsh_c['length'] * s, axis=lsh_c['axis'])
    ax.plot_surface(lsh_x, lsh_y, lsh_z, color='#4A90E2', alpha=0.9)

    # Right leg
    rth_c = p['right_thigh']
    right_thigh_center = tuple(np.array(rth_c['center']) * np.array([s, s, s]))
    rth_x, rth_y, rth_z = create_cylinder(right_thigh_center, rth_c['radius'] * s, rth_c['length'] * s, axis=rth_c['axis'])
    ax.plot_surface(rth_x, rth_y, rth_z, color='#4A90E2', alpha=0.9)
    rsh_c = p['right_shin']
    right_shin_center = tuple(np.array(rsh_c['center']) * np.array([s, s, s]))
    rsh_x, rsh_y, rsh_z = create_cylinder(right_shin_center, rsh_c['radius'] * s, rsh_c['length'] * s, axis=rsh_c['axis'])
    ax.plot_surface(rsh_x, rsh_y, rsh_z, color='#4A90E2', alpha=0.9)
    
    # Hands (small spheres)
    hand_radius = p['hands']['radius'] * s
    
    # Left hand
    left_hand_center = tuple(np.array(p['hands']['left_center']) * np.array([s, s, s]))
    left_hand_x, left_hand_y, left_hand_z = create_sphere(left_hand_center, hand_radius, resolution=10)
    ax.plot_surface(left_hand_x, left_hand_y, left_hand_z, color='#D2B48C', alpha=0.9)
    
    # Right hand
    right_hand_center = tuple(np.array(p['hands']['right_center']) * np.array([s, s, s]))
    right_hand_x, right_hand_y, right_hand_z = create_sphere(right_hand_center, hand_radius, resolution=10)
    ax.plot_surface(right_hand_x, right_hand_y, right_hand_z, color='#D2B48C', alpha=0.9)
    
    # Feet (small boxes)
    foot_size = tuple(np.array(p['feet']['size']) * s)
    
    # Left foot
    left_foot_center = tuple(np.array(p['feet']['left_center']) * np.array([s, s, s]))
    left_foot_faces = create_box(left_foot_center, *foot_size)
    left_foot_collection = Poly3DCollection(left_foot_faces, facecolors='#2C3E50', alpha=0.9, edgecolors='black')
    ax.add_collection3d(left_foot_collection)
    
    # Right foot
    right_foot_center = tuple(np.array(p['feet']['right_center']) * np.array([s, s, s]))
    right_foot_faces = create_box(right_foot_center, *foot_size)
    right_foot_collection = Poly3DCollection(right_foot_faces, facecolors='#2C3E50', alpha=0.9, edgecolors='black')
    ax.add_collection3d(right_foot_collection)
    
    # Set the aspect ratio and limits
    ax.set_xlim([-p['limits']['x'] * s, p['limits']['x'] * s])
    ax.set_ylim([-p['limits']['y'] * s, p['limits']['y'] * s])
    ax.set_zlim([0, p['limits']['z'] * s])
    
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
    # Create and display the geometric human model with defaults
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