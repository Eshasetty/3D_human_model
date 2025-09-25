"""
Main module for 3D human body visualization.
This is the entry point for the application.
"""

from gui import create_camera_control_gui

def main():
    """Main entry point for the 3D human body visualization application."""
    print("Starting 3D Human Body Visualization...")
    print("Use the sliders to control camera position and click 'Take Snapshot' to capture images.")
    
    # Create and show the GUI
    create_camera_control_gui()

if __name__ == "__main__":
    main()