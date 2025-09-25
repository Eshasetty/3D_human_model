import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.widgets import Slider, Button

class HumanBodyCamera:
    def __init__(self, image_width=800, image_height=600):
        # ASSIGN VALUES HERE when creating the camera
        self.alpha = 500    # Your choice - moderate zoom
        self.beta = 500     # Usually same as alpha
        self.x0 = image_width / 2
        self.y0 = image_height / 2
        self.image_width = image_width
        self.image_height = image_height
        
        # This intrinsic matrix is now FIXED for this camera
        self.K = self._build_intrinsic_matrix()
    
    def _build_intrinsic_matrix(self):
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
            
        right = _normalize(np.cross(forward, world_up))
        up = _normalize(np.cross(right, forward))
        
        # Camera coordinate frame: x=right, y=up, z=forward (into scene is negative z)
        R = np.vstack([right, up, -forward])
        t = -R @ np.array(camera_position)
        return R, t
    
    def project_points(self, world_points, camera_position, camera_direction):
        """Project 3D world points to 2D image coordinates using perspective projection."""
        R, t = self._calculate_extrinsics(camera_position, camera_direction)
        M = self.K @ np.hstack([R, t.reshape(-1, 1)])
        
        # Convert to homogeneous coordinates
        if world_points.shape[1] == 3:
            world_homo = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
        else:
            world_homo = world_points
            
        # Project
        image_homo = (M @ world_homo.T).T
        
        # Convert back to 2D (divide by z)
        z = image_homo[:, 2]
        z[abs(z) < 1e-9] = 1e-9  # avoid division by zero
        image_2d = image_homo[:, :2] / z[:, np.newaxis]
        
        return image_2d

def create_camera_control_gui(params=None):
    """Create a matplotlib-based GUI with camera controls and 3D visualization."""
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main 3D plot (left side)
    ax = fig.add_subplot(121, projection='3d')
    ax.mouse_init()
    
    # Control panel (right side)
    control_ax = fig.add_subplot(122)
    control_ax.set_xlim(0, 1)
    control_ax.set_ylim(0, 1)
    control_ax.axis('off')
    
    # Set background colors
    fig.patch.set_facecolor('#6B5B95')
    ax.set_facecolor('#4A4A4A')
    
    # Merge params with defaults
    p = deep_update(get_default_params(), params or {})
    s = float(p['scale'])
    
    # Initialize camera position
    cam_pos = np.array(p['camera']['position']) * s
    state = {'cam_pos': cam_pos, 'scale': s, 'params': p}
    
    # Create body camera
    body_camera = HumanBodyCamera(image_width=800, image_height=600)
    state['body_camera'] = body_camera
    
    # Build the 3D scene
    def build_scene():
        ax.clear()
        ax.set_facecolor('#4A4A4A')
        
        # Ground plane
        add_ground_plane(
            ax,
            size=p['ground']['size'] * s,
            grid_step=p['ground']['grid_step'] * s,
            z=p['ground']['z'],
            color=p['ground']['color'],
            alpha=p['ground']['alpha'],
            grid_color=p['ground']['grid_color'],
            grid_alpha=p['ground']['grid_alpha']
        )
        
        # Head
        head_center = tuple(np.array(p['head']['center']) * np.array([s, s, s]))
        head_radius = p['head']['radius'] * s
        head_x, head_y, head_z = create_sphere(head_center, head_radius)
        ax.plot_surface(head_x, head_y, head_z, color='#D2B48C', alpha=0.9)
        
        # Torso
        torso_center = tuple(np.array(p['torso']['center']) * np.array([s, s, s]))
        torso_faces = create_box(
            torso_center,
            p['torso']['width'] * s,
            p['torso']['height'] * s,
            p['torso']['depth'] * s
        )
        torso_collection = Poly3DCollection(torso_faces, facecolors='#4A90E2', alpha=0.9, edgecolors='black')
        ax.add_collection3d(torso_collection)
        
        # Arms and legs (simplified for GUI)
        # Left arm
        lua_c = p['left_upper_arm']
        left_upper_arm_center = tuple(np.array(lua_c['center']) * np.array([s, s, s]))
        lua_x, lua_y, lua_z = create_cylinder(left_upper_arm_center, lua_c['radius'] * s, lua_c['length'] * s, axis=lua_c['axis'])
        ax.plot_surface(lua_x, lua_y, lua_z, color='#4A90E2', alpha=0.9)
        
        # Right arm
        rua_c = p['right_upper_arm']
        right_upper_arm_center = tuple(np.array(rua_c['center']) * np.array([s, s, s]))
        rua_x, rua_y, rua_z = create_cylinder(right_upper_arm_center, rua_c['radius'] * s, rua_c['length'] * s, axis=rua_c['axis'])
        ax.plot_surface(rua_x, rua_y, rua_z, color='#4A90E2', alpha=0.9)
        
        # Left leg
        lth_c = p['left_thigh']
        left_thigh_center = tuple(np.array(lth_c['center']) * np.array([s, s, s]))
        lth_x, lth_y, lth_z = create_cylinder(left_thigh_center, lth_c['radius'] * s, lth_c['length'] * s, axis=lth_c['axis'])
        ax.plot_surface(lth_x, lth_y, lth_z, color='#4A90E2', alpha=0.9)
        
        # Right leg
        rth_c = p['right_thigh']
        right_thigh_center = tuple(np.array(rth_c['center']) * np.array([s, s, s]))
        rth_x, rth_y, rth_z = create_cylinder(right_thigh_center, rth_c['radius'] * s, rth_c['length'] * s, axis=rth_c['axis'])
        ax.plot_surface(rth_x, rth_y, rth_z, color='#4A90E2', alpha=0.9)
        
        # Camera
        cam = p['camera']
        cam_size = tuple(np.array(cam['size']) * s)
        camera_art = draw_camera(ax, tuple(state['cam_pos']), cam_size, cam['color'], cam['lens_radius'] * s, picker_size=cam['picker_size'])
        state['camera_art'] = camera_art
        
        # Set limits
        ax.set_xlim([-p['limits']['x'] * s, p['limits']['x'] * s])
        ax.set_ylim([-p['limits']['y'] * s, p['limits']['y'] * s])
        ax.set_zlim([0, p['limits']['z'] * s])
        
        ax.set_axis_off()
        ax.view_init(elev=10, azim=45)
        
        return torso_center
    
    # Build initial scene
    torso_center = build_scene()
    state['torso_center'] = torso_center
    
    # Create sliders
    ax_x = plt.axes([0.65, 0.8, 0.3, 0.03])
    ax_y = plt.axes([0.65, 0.75, 0.3, 0.03])
    ax_z = plt.axes([0.65, 0.7, 0.3, 0.03])
    
    slider_x = Slider(ax_x, 'Camera X', -3, 3, valinit=state['cam_pos'][0], valfmt='%.1f')
    slider_y = Slider(ax_y, 'Camera Y', -3, 3, valinit=state['cam_pos'][1], valfmt='%.1f')
    slider_z = Slider(ax_z, 'Camera Z', 0, 4, valinit=state['cam_pos'][2], valfmt='%.1f')
    
    # Create buttons
    ax_left = plt.axes([0.65, 0.55, 0.08, 0.04])
    ax_right = plt.axes([0.75, 0.55, 0.08, 0.04])
    ax_up = plt.axes([0.7, 0.6, 0.08, 0.04])
    ax_down = plt.axes([0.7, 0.5, 0.08, 0.04])
    ax_forward = plt.axes([0.7, 0.55, 0.08, 0.04])
    ax_back = plt.axes([0.8, 0.55, 0.08, 0.04])
    ax_snapshot = plt.axes([0.65, 0.4, 0.3, 0.04])
    
    btn_left = Button(ax_left, 'Left')
    btn_right = Button(ax_right, 'Right')
    btn_up = Button(ax_up, 'Up')
    btn_down = Button(ax_down, 'Down')
    btn_forward = Button(ax_forward, 'Forward')
    btn_back = Button(ax_back, 'Back')
    btn_snapshot = Button(ax_snapshot, 'Take Snapshot')
    
    # Add labels
    control_ax.text(0.5, 0.9, 'Camera Controls', fontsize=16, ha='center', weight='bold')
    control_ax.text(0.5, 0.85, 'Use sliders to control camera position', fontsize=12, ha='center')
    control_ax.text(0.5, 0.45, 'Quick Movement Buttons', fontsize=14, ha='center', weight='bold')
    
    # Update functions
    def update_camera(val=None):
        state['cam_pos'] = np.array([slider_x.val, slider_y.val, slider_z.val])
        build_scene()
        fig.canvas.draw()
    
    def move_camera(dx, dy, dz):
        state['cam_pos'] += np.array([dx, dy, dz])
        slider_x.set_val(state['cam_pos'][0])
        slider_y.set_val(state['cam_pos'][1])
        slider_z.set_val(state['cam_pos'][2])
        update_camera()
    
    def take_snapshot(event):
        # Use the HumanBodyCamera to take a proper snapshot with full geometry
        cam_pos = tuple(state['cam_pos'])
        target = state['torso_center']
        camera_direction = np.array(target) - np.array(cam_pos)
        
        # Create offscreen figure
        fig2 = Figure(figsize=(8, 6))
        FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(0, 800)
        ax2.set_ylim(0, 600)
        ax2.invert_yaxis()
        ax2.axis('off')
        
        camera = state['body_camera']
        
        # Project all body parts using the camera
        def project_and_draw_surface(X, Y, Z, color, alpha=0.8):
            # Sample points from the surface
            points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            projected = camera.project_points(points, cam_pos, camera_direction)
            
            # Filter valid points (within image bounds and in front of camera)
            valid_mask = (projected[:, 0] >= 0) & (projected[:, 0] <= 800) & \
                        (projected[:, 1] >= 0) & (projected[:, 1] <= 600)
            valid_pts = projected[valid_mask]
            
            if len(valid_pts) > 10:  # Need enough points to form a surface
                # Create a scatter plot to represent the surface
                ax2.scatter(valid_pts[:, 0], valid_pts[:, 1], c=color, s=8, alpha=alpha)
        
        def project_and_draw_box(faces, color, alpha=0.8):
            for face in faces:
                points = np.array(face)
                projected = camera.project_points(points, cam_pos, camera_direction)
                
                # Filter valid points
                valid_mask = (projected[:, 0] >= 0) & (projected[:, 0] <= 800) & \
                            (projected[:, 1] >= 0) & (projected[:, 1] <= 600)
                valid_pts = projected[valid_mask]
                
                if len(valid_pts) > 2:  # Need at least 3 points for a polygon
                    from matplotlib.patches import Polygon
                    poly = Polygon(valid_pts, closed=True, facecolor=color, alpha=alpha, 
                                 edgecolor='black', linewidth=0.5)
                    ax2.add_patch(poly)
        
        # Recreate the full body geometry for projection
        # Head
        head_center = tuple(np.array(p['head']['center']) * np.array([s, s, s]))
        head_radius = p['head']['radius'] * s
        head_x, head_y, head_z = create_sphere(head_center, head_radius)
        project_and_draw_surface(head_x, head_y, head_z, '#D2B48C', 0.9)
        
        # Torso
        torso_center = tuple(np.array(p['torso']['center']) * np.array([s, s, s]))
        torso_faces = create_box(
            torso_center,
            p['torso']['width'] * s,
            p['torso']['height'] * s,
            p['torso']['depth'] * s
        )
        project_and_draw_box(torso_faces, '#4A90E2', 0.9)
        
        # Arms
        lua_c = p['left_upper_arm']
        left_upper_arm_center = tuple(np.array(lua_c['center']) * np.array([s, s, s]))
        lua_x, lua_y, lua_z = create_cylinder(left_upper_arm_center, lua_c['radius'] * s, lua_c['length'] * s, axis=lua_c['axis'])
        project_and_draw_surface(lua_x, lua_y, lua_z, '#4A90E2', 0.9)
        
        lfa_c = p['left_forearm']
        left_forearm_center = tuple(np.array(lfa_c['center']) * np.array([s, s, s]))
        lfa_x, lfa_y, lfa_z = create_cylinder(left_forearm_center, lfa_c['radius'] * s, lfa_c['length'] * s, axis=lfa_c['axis'])
        project_and_draw_surface(lfa_x, lfa_y, lfa_z, '#4A90E2', 0.9)
        
        rua_c = p['right_upper_arm']
        right_upper_arm_center = tuple(np.array(rua_c['center']) * np.array([s, s, s]))
        rua_x, rua_y, rua_z = create_cylinder(right_upper_arm_center, rua_c['radius'] * s, rua_c['length'] * s, axis=rua_c['axis'])
        project_and_draw_surface(rua_x, rua_y, rua_z, '#4A90E2', 0.9)
        
        rfa_c = p['right_forearm']
        right_forearm_center = tuple(np.array(rfa_c['center']) * np.array([s, s, s]))
        rfa_x, rfa_y, rfa_z = create_cylinder(right_forearm_center, rfa_c['radius'] * s, rfa_c['length'] * s, axis=rfa_c['axis'])
        project_and_draw_surface(rfa_x, rfa_y, rfa_z, '#4A90E2', 0.9)
        
        # Legs
        lth_c = p['left_thigh']
        left_thigh_center = tuple(np.array(lth_c['center']) * np.array([s, s, s]))
        lth_x, lth_y, lth_z = create_cylinder(left_thigh_center, lth_c['radius'] * s, lth_c['length'] * s, axis=lth_c['axis'])
        project_and_draw_surface(lth_x, lth_y, lth_z, '#4A90E2', 0.9)
        
        lsh_c = p['left_shin']
        left_shin_center = tuple(np.array(lsh_c['center']) * np.array([s, s, s]))
        lsh_x, lsh_y, lsh_z = create_cylinder(left_shin_center, lsh_c['radius'] * s, lsh_c['length'] * s, axis=lsh_c['axis'])
        project_and_draw_surface(lsh_x, lsh_y, lsh_z, '#4A90E2', 0.9)
        
        rth_c = p['right_thigh']
        right_thigh_center = tuple(np.array(rth_c['center']) * np.array([s, s, s]))
        rth_x, rth_y, rth_z = create_cylinder(right_thigh_center, rth_c['radius'] * s, rth_c['length'] * s, axis=rth_c['axis'])
        project_and_draw_surface(rth_x, rth_y, rth_z, '#4A90E2', 0.9)
        
        rsh_c = p['right_shin']
        right_shin_center = tuple(np.array(rsh_c['center']) * np.array([s, s, s]))
        rsh_x, rsh_y, rsh_z = create_cylinder(right_shin_center, rsh_c['radius'] * s, rsh_c['length'] * s, axis=rsh_c['axis'])
        project_and_draw_surface(rsh_x, rsh_y, rsh_z, '#4A90E2', 0.9)
        
        # Hands
        hand_radius = p['hands']['radius'] * s
        left_hand_center = tuple(np.array(p['hands']['left_center']) * np.array([s, s, s]))
        left_hand_x, left_hand_y, left_hand_z = create_sphere(left_hand_center, hand_radius)
        project_and_draw_surface(left_hand_x, left_hand_y, left_hand_z, '#D2B48C', 0.9)
        
        right_hand_center = tuple(np.array(p['hands']['right_center']) * np.array([s, s, s]))
        right_hand_x, right_hand_y, right_hand_z = create_sphere(right_hand_center, hand_radius)
        project_and_draw_surface(right_hand_x, right_hand_y, right_hand_z, '#D2B48C', 0.9)
        
        # Feet
        foot_size = tuple(np.array(p['feet']['size']) * s)
        left_foot_center = tuple(np.array(p['feet']['left_center']) * np.array([s, s, s]))
        left_foot_faces = create_box(left_foot_center, *foot_size)
        project_and_draw_box(left_foot_faces, '#2C3E50', 0.9)
        
        right_foot_center = tuple(np.array(p['feet']['right_center']) * np.array([s, s, s]))
        right_foot_faces = create_box(right_foot_center, *foot_size)
        project_and_draw_box(right_foot_faces, '#2C3E50', 0.9)
        
        fig2.savefig('snapshot_from_camera.png', dpi=220, bbox_inches='tight')
        print('Saved snapshot_from_camera.png from camera POV with full geometry')
    
    # Bind events
    slider_x.on_changed(update_camera)
    slider_y.on_changed(update_camera)
    slider_z.on_changed(update_camera)
    
    btn_left.on_clicked(lambda event: move_camera(-0.2, 0, 0))
    btn_right.on_clicked(lambda event: move_camera(0.2, 0, 0))
    btn_up.on_clicked(lambda event: move_camera(0, 0.2, 0))
    btn_down.on_clicked(lambda event: move_camera(0, -0.2, 0))
    btn_forward.on_clicked(lambda event: move_camera(0, 0, 0.2))
    btn_back.on_clicked(lambda event: move_camera(0, 0, -0.2))
    btn_snapshot.on_clicked(take_snapshot)
    
    plt.tight_layout()
    plt.show()

def get_default_params():
    """Return default per-part parameters for the geometric human model."""
    return {
        'scale': 1.0,
        'ground': {
            'size': 2.5,
            'grid_step': 0.25,
            'z': 0.0,
            'color': '#2C2F3A',
            'alpha': 0.35,
            'grid_color': '#3E4454',
            'grid_alpha': 0.25
        },
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
        'camera': {
            'position': (1.0, 0.6, 0.12),
            'size': (0.12, 0.08, 0.06),
            'color': '#202225',
            'lens_radius': 0.02,
            'move_step': 0.12,
            'picker_size': 100
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

def add_ground_plane(ax, size=2.5, grid_step=0.25, z=0.0, color='#2C2F3A', alpha=0.35, grid_color='#3E4454', grid_alpha=0.25):
    """Add a flat ground plane with a subtle grid at height z."""
    half = size
    plane_vertices = np.array([
        [-half, -half, z],
        [ half, -half, z],
        [ half,  half, z],
        [-half,  half, z]
    ])
    plane_face = [plane_vertices[0], plane_vertices[1], plane_vertices[2], plane_vertices[3]]

    plane = Poly3DCollection([plane_face], facecolors=color, alpha=alpha, edgecolors='none')
    ax.add_collection3d(plane)

    # draw grid lines
    ticks = np.arange(-half, half + 1e-6, grid_step)
    for t in ticks:
        ax.plot([-half, half], [t, t], [z, z], color=grid_color, alpha=grid_alpha, linewidth=0.8)
        ax.plot([t, t], [-half, half], [z, z], color=grid_color, alpha=grid_alpha, linewidth=0.8)

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

def draw_camera(ax, position, size, color, lens_radius, picker_size=100):
    """Draw a small camera as a box with a small lens cylinder; return artists."""
    cam_faces = create_box(position, *size)
    cam_box = Poly3DCollection(cam_faces, facecolors=color, alpha=0.9, edgecolors='black', linewidths=0.5)
    ax.add_collection3d(cam_box)

    # lens: small cylinder on the front face, pointing along +y
    lens_center = (position[0], position[1] + size[1]/2 + lens_radius/2, position[2])
    lx, ly, lz = create_cylinder(lens_center, lens_radius, height=lens_radius*1.2, axis='y')
    lens = ax.plot_surface(lx, ly, lz, color='#4A90E2', alpha=0.9, picker=picker_size)

    return {'box': cam_box, 'lens': lens}

def compute_view_from_camera(camera_pos, target_pos):
    """Compute (elev, azim) to look from camera_pos toward target_pos for mplot3d.

    Approximates Matplotlib's angle conventions: elev is degrees above the xy-plane,
    azim rotates around z where azim=0 looks toward -y. We map using arctan2.
    """
    v = np.array(target_pos, dtype=float) - np.array(camera_pos, dtype=float)
    xy = np.hypot(v[0], v[1])
    r = np.hypot(xy, v[2]) + 1e-9
    elev = np.degrees(np.arcsin(v[2] / r))
    azim = np.degrees(np.arctan2(v[0], v[1]))  # map x/y into azim convention
    return {'elev': float(elev), 'azim': float(azim)}

def center_axes_on_target(ax, center, span):
    """Center the axes limits on a given 3D point with a cubic span."""
    cx, cy, cz = center
    half = span / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

def _normalize(vec):
    v = np.array(vec, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def compute_camera_transform(cam_pos, forward=(0, 1, 0), world_up=(0, 0, 1)):
    """Return rotation R (3x3) and translation t for world->camera coordinates.

    Camera at cam_pos, looking along `forward` in world coordinates. The camera
    coordinate frame uses: x=right, y=up, z=forward (into scene is negative z).
    We construct R so that p_cam = R @ (p_world - cam_pos).
    """
    f = _normalize(forward)
    up = _normalize(world_up)
    # If forward is nearly parallel to up, pick a different up
    if abs(np.dot(f, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    right = _normalize(np.cross(f, up))
    up_cam = _normalize(np.cross(right, f))
    R = np.vstack([right, up_cam, -f])  # rows are basis vectors
    t = -R @ np.array(cam_pos, dtype=float)
    return R, t

def transform_surface(X, Y, Z, R, t):
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pts_cam = (R @ pts.T).T + t
    Xc = pts_cam[:, 0].reshape(X.shape)
    Yc = pts_cam[:, 1].reshape(Y.shape)
    Zc = pts_cam[:, 2].reshape(Z.shape)
    return Xc, Yc, Zc

def transform_faces(faces, R, t):
    out = []
    for face in faces:
        pts = np.array(face)
        pts_cam = (R @ pts.T).T + t
        out.append([tuple(p) for p in pts_cam])
    return out

def surface_to_quads(X, Y, Z):
    """Convert a rectilinear surface grid to a list of quad polygons with depth.

    Returns a list of tuples: (mean_z, [(x,y), ... 4 pts])
    """
    quads = []
    ni, nj = X.shape
    for i in range(ni - 1):
        for j in range(nj - 1):
            xs = [X[i, j], X[i+1, j], X[i+1, j+1], X[i, j+1]]
            ys = [Y[i, j], Y[i+1, j], Y[i+1, j+1], Y[i, j+1]]
            zs = [Z[i, j], Z[i+1, j], Z[i+1, j+1], Z[i, j+1]]
            quads.append((float(np.mean(zs)), list(zip(xs, ys))))
    return quads

def plot_geometric_human(params: dict | None = None, include_camera: bool = True, show: bool = True, set_view: dict | None = None):
    """Create and plot a geometric human model using per-part parameters.

    params: optional dict overriding defaults from `get_default_params()`.
    Supports a global 'scale' applied to all linear dimensions.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.mouse_init()  # ensure drag-to-rotate and scroll-to-zoom are active
    
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
        z=p['ground']['z'],
        color=p['ground']['color'],
        alpha=p['ground']['alpha'],
        grid_color=p['ground']['grid_color'],
        grid_alpha=p['ground']['grid_alpha']
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
    
    # Camera model
    cam = p['camera']
    cam_pos = tuple(np.array(cam['position']) * np.array([s, s, s]))
    cam_size = tuple(np.array(cam['size']) * s)
    camera_art = draw_camera(ax, cam_pos, cam_size, cam['color'], cam['lens_radius'] * s, picker_size=cam['picker_size'])

    # Create HumanBodyCamera instance
    body_camera = HumanBodyCamera(image_width=800, image_height=600)

    # Interactivity: click lens or press 'c' to snapshot; move camera in 3D with keys
    state = {'cam_pos': np.array(cam_pos, dtype=float), 'body_camera': body_camera}

    def snapshot():
        # Use HumanBodyCamera for proper perspective projection
        cam_pos = tuple(state['cam_pos'])
        target = torso_center
        camera_direction = np.array(target) - np.array(cam_pos)

        # Use offscreen Agg figure/canvas to avoid interfering with the main window
        fig2 = Figure(figsize=(8, 6))
        FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(0, 800)
        ax2.set_ylim(0, 600)
        ax2.invert_yaxis()  # Image coordinates: y increases downward
        ax2.axis('off')

        # Collect all 3D points from body parts and project using HumanBodyCamera
        camera = state['body_camera']
        
        # Sample points from surfaces and project them
        def sample_and_project_surface(X, Y, Z, color, sample_rate=4):
            # Sample every nth point to reduce density
            sampled_x = X[::sample_rate, ::sample_rate]
            sampled_y = Y[::sample_rate, ::sample_rate]
            sampled_z = Z[::sample_rate, ::sample_rate]
            
            points = np.stack([sampled_x.ravel(), sampled_y.ravel(), sampled_z.ravel()], axis=1)
            projected = camera.project_points(points, cam_pos, camera_direction)
            
            # Filter valid points (within image bounds and in front of camera)
            valid_mask = (projected[:, 0] >= 0) & (projected[:, 0] <= 800) & \
                        (projected[:, 1] >= 0) & (projected[:, 1] <= 600)
            valid_pts = projected[valid_mask]
            
            if len(valid_pts) > 0:
                ax2.scatter(valid_pts[:, 0], valid_pts[:, 1], c=color, s=8, alpha=0.7)
        
        def project_box_faces(faces, color):
            for face in faces:
                points = np.array(face)
                projected = camera.project_points(points, cam_pos, camera_direction)
                # Filter valid points
                valid_mask = (projected[:, 0] >= 0) & (projected[:, 0] <= 800) & \
                            (projected[:, 1] >= 0) & (projected[:, 1] <= 600)
                valid_pts = projected[valid_mask]
                
                if len(valid_pts) > 2:  # Need at least 3 points for a triangle
                    from matplotlib.patches import Polygon
                    poly = Polygon(valid_pts, closed=True, facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                    ax2.add_patch(poly)

        # Project and draw all body parts using HumanBodyCamera
        sample_and_project_surface(head_x, head_y, head_z, color='#D2B48C', sample_rate=2)
        project_box_faces(torso_faces, '#4A90E2')
        sample_and_project_surface(lua_x, lua_y, lua_z, color='#4A90E2')
        sample_and_project_surface(lfa_x, lfa_y, lfa_z, color='#4A90E2')
        sample_and_project_surface(rua_x, rua_y, rua_z, color='#4A90E2')
        sample_and_project_surface(rfa_x, rfa_y, rfa_z, color='#4A90E2')
        sample_and_project_surface(lth_x, lth_y, lth_z, color='#4A90E2')
        sample_and_project_surface(lsh_x, lsh_y, lsh_z, color='#4A90E2')
        sample_and_project_surface(rth_x, rth_y, rth_z, color='#4A90E2')
        sample_and_project_surface(rsh_x, rsh_y, rsh_z, color='#4A90E2')
        # hands and feet
        sample_and_project_surface(left_hand_x, left_hand_y, left_hand_z, color='#D2B48C', sample_rate=2)
        sample_and_project_surface(right_hand_x, right_hand_y, right_hand_z, color='#D2B48C', sample_rate=2)
        project_box_faces(left_foot_faces, '#2C3E50')
        project_box_faces(right_foot_faces, '#2C3E50')

        fig2.savefig('snapshot_from_camera.png', dpi=220, bbox_inches='tight')
        print('Saved snapshot_from_camera.png from camera POV using HumanBodyCamera')

    def on_pick(event):
        if event.artist is camera_art['lens']:
            # Save 2D snapshot from current axes view as a simple effect
            snapshot()

    def on_key(event):
        step = cam['move_step'] * s
        moved = False
        # X axis
        if event.key in ('left', 'a'):
            state['cam_pos'][0] -= step
            moved = True
        elif event.key in ('right', 'd'):
            state['cam_pos'][0] += step
            moved = True
        # Y axis
        elif event.key in ('up', 'w'):
            state['cam_pos'][1] += step
            moved = True
        elif event.key in ('down', 's'):
            state['cam_pos'][1] -= step
            moved = True
        # Z axis
        elif event.key in ('pageup', 'q'):
            state['cam_pos'][2] += step
            moved = True
        elif event.key in ('pagedown', 'e'):
            state['cam_pos'][2] -= step
            moved = True
        # Snapshot shortcut
        elif event.key == 'c':
            snapshot()
            return
        if moved:
            # Redraw camera at new position
            camera_art['box'].remove()
            camera_art['lens'].remove()
            new_art = draw_camera(
                ax,
                tuple(state['cam_pos']),
                cam_size,
                cam['color'],
                cam['lens_radius'] * s,
                picker_size=cam['picker_size']
            )
            camera_art.update(new_art)
            plt.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Set the aspect ratio and limits
    ax.set_xlim([-p['limits']['x'] * s, p['limits']['x'] * s])
    ax.set_ylim([-p['limits']['y'] * s, p['limits']['y'] * s])
    ax.set_zlim([0, p['limits']['z'] * s])
    
    # Set title on the figure (keep visible when axes are hidden)
    fig.suptitle('Geometric Human Model', fontsize=16, color='white')
    
    # Remove axes for a cleaner look
    ax.set_axis_off()
    
    # Set viewing angle
    if set_view is not None:
        ax.view_init(elev=set_view.get('elev', 10), azim=set_view.get('azim', 45))
    else:
        ax.view_init(elev=10, azim=45)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax

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
    # Launch the GUI with camera controls
    create_camera_control_gui()
    
    print("\n3D Human Model with Camera Controls!")
    print("- Use sliders to control camera X, Y, Z position")
    print("- Use directional buttons for quick movement")
    print("- Click 'Take Snapshot' to capture from camera POV")
    print("- Drag to rotate the 3D view, scroll to zoom")