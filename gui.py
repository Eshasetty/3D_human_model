"""
GUI module for 3D human body visualization.
Contains GUI controls, layout, and interaction handling.
"""

import numpy as np
import matplotlib
# Don't set backend - use default like working version
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from camera import HumanBodyCamera
from geometry import create_sphere, create_cylinder, create_box, add_ground_plane
from config import get_default_params, deep_update


def create_camera_control_gui(params=None):
    """Create a matplotlib-based GUI with camera controls and 3D visualization."""
    
    # Initialize state
    state = {
        'cam_pos': np.array([0.0, -1.5, 0.8]),
        'torso_center': np.array([0.0, 0.0, 1.1]),
        'body_camera': HumanBodyCamera(),
        'params': deep_update(get_default_params(), params or {})
    }
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 3D visualization subplot
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.mouse_init()  # Enable 3D navigation
    
    # Control panel subplot
    control_ax = fig.add_subplot(122)
    control_ax.set_xlim(0, 1)
    control_ax.set_ylim(0, 1)
    control_ax.axis('off')
    
    # Set background colors
    fig.patch.set_facecolor('#6B5B95')
    ax_3d.set_facecolor('#4A4A4A')
    
    def build_scene():
        """Build the 3D scene with human body and camera."""
        ax_3d.clear()
        ax_3d.mouse_init()
        
        p = state['params']
        s = float(p['scale'])
        
        # Ground plane
        add_ground_plane(
            ax_3d,
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
        ax_3d.plot_surface(head_x, head_y, head_z, color='#D2B48C', alpha=0.9)
        
        # Torso
        torso_center = tuple(np.array(p['torso']['center']) * np.array([s, s, s]))
        torso_faces = create_box(
            torso_center,
            p['torso']['width'] * s,
            p['torso']['height'] * s,
            p['torso']['depth'] * s
        )
        torso_collection = Poly3DCollection(torso_faces, facecolors='#4A90E2', alpha=0.9, edgecolors='black')
        ax_3d.add_collection3d(torso_collection)
        
        # Arms and legs
        _add_limbs(ax_3d, p, s)
        
        # Camera model
        _draw_camera(ax_3d, state['cam_pos'], p['camera'], s)
        
        # Set view limits
        ax_3d.set_xlim(-p['limits']['x'], p['limits']['x'])
        ax_3d.set_ylim(-p['limits']['y'], p['limits']['y'])
        ax_3d.set_zlim(0, p['limits']['z'])
        
        # Remove axes to make it look like standing on a clean surface
        ax_3d.set_axis_off()
        ax_3d.set_title('3D Human Body Model', pad=20)
    
    def _add_limbs(ax, p, s):
        """Add arms and legs to the scene."""
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
        
        # Feet
        foot_size = tuple(np.array(p['feet']['size']) * s)
        left_foot_center = tuple(np.array(p['feet']['left_center']) * np.array([s, s, s]))
        left_foot_faces = create_box(left_foot_center, *foot_size)
        left_foot_collection = Poly3DCollection(left_foot_faces, facecolors='#2C3E50', alpha=0.9, edgecolors='black')
        ax.add_collection3d(left_foot_collection)
        
        right_foot_center = tuple(np.array(p['feet']['right_center']) * np.array([s, s, s]))
        right_foot_faces = create_box(right_foot_center, *foot_size)
        right_foot_collection = Poly3DCollection(right_foot_faces, facecolors='#2C3E50', alpha=0.9, edgecolors='black')
        ax.add_collection3d(right_foot_collection)
    
    def _draw_camera(ax, position, camera_params, scale):
        """Draw the camera model in the 3D scene."""
        cam_size = tuple(np.array(camera_params['size']) * scale)
        cam_faces = create_box(position, *cam_size)
        cam_collection = Poly3DCollection(cam_faces, facecolors=camera_params['color'], alpha=0.9, edgecolors='black')
        ax.add_collection3d(cam_collection)
        
        # Camera lens
        lens_center = (position[0], position[1] + cam_size[1]/2, position[2])
        lens_radius = camera_params['lens_radius'] * scale
        lens_x, lens_y, lens_z = create_sphere(lens_center, lens_radius)
        ax.plot_surface(lens_x, lens_y, lens_z, color='#FFD700', alpha=0.9)
    
    # Create sliders (moved to right side)
    ax_x = plt.axes([0.65, 0.7, 0.3, 0.03])
    ax_y = plt.axes([0.65, 0.6, 0.3, 0.03])
    ax_z = plt.axes([0.65, 0.5, 0.3, 0.03])
    
    slider_x = Slider(ax_x, 'Camera X', -2.0, 2.0, valinit=state['cam_pos'][0], valfmt='%.2f')
    slider_y = Slider(ax_y, 'Camera Y', -2.0, 2.0, valinit=state['cam_pos'][1], valfmt='%.2f')
    slider_z = Slider(ax_z, 'Camera Z', 0.0, 2.0, valinit=state['cam_pos'][2], valfmt='%.2f')
    
    # Make sliders visible and interactive
    ax_x.set_facecolor('lightgray')
    ax_y.set_facecolor('lightgray')
    ax_z.set_facecolor('lightgray')
    
    # Ensure sliders are properly initialized
    slider_x.set_active(True)
    slider_y.set_active(True)
    slider_z.set_active(True)
    
    # Create snapshot button
    ax_snapshot = plt.axes([0.65, 0.4, 0.3, 0.04])
    btn_snapshot = Button(ax_snapshot, 'Take Snapshot')
    
    # Add labels
    control_ax.text(0.5, 0.9, 'Camera Controls', fontsize=16, ha='center', weight='bold')
    control_ax.text(0.5, 0.45, 'Take a snapshot from the camera\'s perspective', fontsize=12, ha='center')
    
    # Update functions
    def update_camera(val=None):
        state['cam_pos'] = np.array([slider_x.val, slider_y.val, slider_z.val])
        print(f"Camera moved to: {state['cam_pos']}")
        build_scene()
        fig.canvas.draw_idle()
    
    
    def take_snapshot(event):
        """Take a snapshot from the camera's perspective."""
        cam_pos = tuple(state['cam_pos'])
        target = state['torso_center']
        camera_direction = np.array(target) - np.array(cam_pos)
        
        # Create offscreen figure
        fig2 = Figure(figsize=(8, 6))
        FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(0, 800)
        ax2.set_ylim(600, 0)
        ax2.axis('off')
        
        camera = state['body_camera']
        all_patches = []
        
        def project_and_draw_surface(X, Y, Z, color, alpha=0.8):
            points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            projected = camera.project_points(points, cam_pos, camera_direction)
            
            valid_mask = (projected[:, 0] >= 0) & (projected[:, 0] <= 800) & \
                        (projected[:, 1] >= 0) & (projected[:, 1] <= 600)
            valid_pts = projected[valid_mask]
            
            if len(valid_pts) > 20:
                from matplotlib.patches import Polygon
                
                def simple_convex_hull(points):
                    if len(points) < 3:
                        return points
                    leftmost = np.argmin(points[:, 0])
                    hull = [leftmost]
                    
                    while True:
                        candidate = 0
                        for i in range(len(points)):
                            if i == hull[-1]:
                                continue
                            o1 = points[hull[-1]]
                            o2 = points[candidate]
                            o3 = points[i]
                            
                            cross = (o2[0] - o1[0]) * (o3[1] - o1[1]) - (o2[1] - o1[1]) * (o3[0] - o1[0])
                            if cross < 0 or (cross == 0 and np.linalg.norm(o3 - o1) > np.linalg.norm(o2 - o1)):
                                candidate = i
                        
                        if candidate == leftmost:
                            break
                        hull.append(candidate)
                    
                    return np.array([points[i] for i in hull])
                
                try:
                    hull_points = simple_convex_hull(valid_pts[:, :2])
                    if len(hull_points) >= 3:
                        polygon = Polygon(hull_points, facecolor=color, alpha=alpha, 
                                         edgecolor='black', linewidth=0.5)
                        depth = np.mean([np.linalg.norm(np.array(cam_pos) - np.array([x, y, z])) 
                                       for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel())])
                        all_patches.append((depth, polygon))
                except:
                    from matplotlib.patches import Ellipse
                    min_x, max_x = np.min(valid_pts[:, 0]), np.max(valid_pts[:, 0])
                    min_y, max_y = np.min(valid_pts[:, 1]), np.max(valid_pts[:, 1])
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    width = max_x - min_x
                    height = max_y - min_y
                    
                    if width > 5 and height > 5:
                        ellipse = Ellipse((center_x, center_y), width, height, 
                                        facecolor=color, alpha=alpha, 
                                        edgecolor='black', linewidth=0.5)
                        depth = np.mean([np.linalg.norm(np.array(cam_pos) - np.array([x, y, z])) 
                                       for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel())])
                        all_patches.append((depth, ellipse))
        
        def project_and_draw_box(faces, color, alpha=0.8):
            for face in faces:
                points = np.array(face)
                projected = camera.project_points(points, cam_pos, camera_direction)
                
                valid_mask = (projected[:, 0] >= 0) & (projected[:, 0] <= 800) & \
                            (projected[:, 1] >= 0) & (projected[:, 1] <= 600)
                valid_pts = projected[valid_mask]
                
                if len(valid_pts) >= 3:
                    from matplotlib.patches import Polygon
                    polygon = Polygon(valid_pts, facecolor=color, alpha=alpha, 
                                   edgecolor='black', linewidth=0.5)
                    depth = np.mean([np.linalg.norm(np.array(cam_pos) - np.array([x, y, z])) 
                                   for x, y, z in face])
                    all_patches.append((depth, polygon))
        
        # Render all body parts
        p = state['params']
        s = float(p['scale'])
        
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
        
        # Arms and legs
        _render_limbs_for_snapshot(p, s, project_and_draw_surface, project_and_draw_box, camera, cam_pos, camera_direction, all_patches)
        
        # Sort and render patches
        all_patches.sort(key=lambda x: x[0], reverse=True)
        for depth, patch in all_patches:
            ax2.add_patch(patch)
        
        fig2.savefig('snapshot_from_camera.png', dpi=220, bbox_inches='tight')
        print('Saved snapshot_from_camera.png from camera POV with full geometry')
    
    def _render_limbs_for_snapshot(p, s, project_and_draw_surface, project_and_draw_box, camera, cam_pos, camera_direction, all_patches):
        """Render limbs for snapshot."""
        # Left arm
        lua_c = p['left_upper_arm']
        left_upper_arm_center = tuple(np.array(lua_c['center']) * np.array([s, s, s]))
        lua_x, lua_y, lua_z = create_cylinder(left_upper_arm_center, lua_c['radius'] * s, lua_c['length'] * s, axis=lua_c['axis'])
        project_and_draw_surface(lua_x, lua_y, lua_z, '#4A90E2', 0.9)
        
        lfa_c = p['left_forearm']
        left_forearm_center = tuple(np.array(lfa_c['center']) * np.array([s, s, s]))
        lfa_x, lfa_y, lfa_z = create_cylinder(left_forearm_center, lfa_c['radius'] * s, lfa_c['length'] * s, axis=lfa_c['axis'])
        project_and_draw_surface(lfa_x, lfa_y, lfa_z, '#4A90E2', 0.9)
        
        # Right arm
        rua_c = p['right_upper_arm']
        right_upper_arm_center = tuple(np.array(rua_c['center']) * np.array([s, s, s]))
        rua_x, rua_y, rua_z = create_cylinder(right_upper_arm_center, rua_c['radius'] * s, rua_c['length'] * s, axis=rua_c['axis'])
        project_and_draw_surface(rua_x, rua_y, rua_z, '#4A90E2', 0.9)
        
        rfa_c = p['right_forearm']
        right_forearm_center = tuple(np.array(rfa_c['center']) * np.array([s, s, s]))
        rfa_x, rfa_y, rfa_z = create_cylinder(right_forearm_center, rfa_c['radius'] * s, rfa_c['length'] * s, axis=rfa_c['axis'])
        project_and_draw_surface(rfa_x, rfa_y, rfa_z, '#4A90E2', 0.9)
        
        # Left leg
        lth_c = p['left_thigh']
        left_thigh_center = tuple(np.array(lth_c['center']) * np.array([s, s, s]))
        lth_x, lth_y, lth_z = create_cylinder(left_thigh_center, lth_c['radius'] * s, lth_c['length'] * s, axis=lth_c['axis'])
        project_and_draw_surface(lth_x, lth_y, lth_z, '#4A90E2', 0.9)
        
        lsh_c = p['left_shin']
        left_shin_center = tuple(np.array(lsh_c['center']) * np.array([s, s, s]))
        lsh_x, lsh_y, lsh_z = create_cylinder(left_shin_center, lsh_c['radius'] * s, lsh_c['length'] * s, axis=lsh_c['axis'])
        project_and_draw_surface(lsh_x, lsh_y, lsh_z, '#4A90E2', 0.9)
        
        # Right leg
        rth_c = p['right_thigh']
        right_thigh_center = tuple(np.array(rth_c['center']) * np.array([s, s, s]))
        rth_x, rth_y, rth_z = create_cylinder(right_thigh_center, rth_c['radius'] * s, rth_c['length'] * s, axis=rth_c['axis'])
        project_and_draw_surface(rth_x, rth_y, rth_z, '#4A90E2', 0.9)
        
        rsh_c = p['right_shin']
        right_shin_center = tuple(np.array(rsh_c['center']) * np.array([s, s, s]))
        rsh_x, rsh_y, rsh_z = create_cylinder(right_shin_center, rsh_c['radius'] * s, rsh_c['length'] * s, axis=rsh_c['axis'])
        project_and_draw_surface(rsh_x, rsh_y, rsh_z, '#4A90E2', 0.9)
        
        # Feet
        foot_size = tuple(np.array(p['feet']['size']) * s)
        left_foot_center = tuple(np.array(p['feet']['left_center']) * np.array([s, s, s]))
        left_foot_faces = create_box(left_foot_center, *foot_size)
        project_and_draw_box(left_foot_faces, '#2C3E50', 0.9)
        
        right_foot_center = tuple(np.array(p['feet']['right_center']) * np.array([s, s, s]))
        right_foot_faces = create_box(right_foot_center, *foot_size)
        project_and_draw_box(right_foot_faces, '#2C3E50', 0.9)
    
    # Bind events
    slider_x.on_changed(update_camera)
    slider_y.on_changed(update_camera)
    slider_z.on_changed(update_camera)
    btn_snapshot.on_clicked(take_snapshot)
    
    # Initial scene
    build_scene()
    
    # Show the GUI
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
