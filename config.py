"""
Configuration module for 3D human body visualization.
Contains default parameters and configuration settings.
"""


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
            'position': (0.0, -1.5, 0.8),
            'size': (0.12, 0.08, 0.06),
            'color': '#202225',
            'lens_radius': 0.02,
            'move_step': 0.12,
            'picker_size': 100
        },
        'joints': {
            'left_arm_angle': 0,
            'right_arm_angle': 0,
            'left_leg_angle': 0,
            'right_leg_angle': 0,
            'left_elbow_angle': 0,
            'right_elbow_angle': 0,
            'left_knee_angle': 0,
            'right_knee_angle': 0
        },
        'limits': {'x': 1.5, 'y': 1.5, 'z': 2.0}
    }


def deep_update(base: dict, overrides: dict) -> dict:
    """
    Recursively update mapping `base` with `overrides` and return it.
    
    Args:
        base: Base dictionary to update
        overrides: Dictionary with override values
        
    Returns:
        Updated dictionary
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
