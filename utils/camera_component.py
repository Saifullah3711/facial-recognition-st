import streamlit as st
import cv2
import time
from utils.camera import Camera

def camera_capture(key_prefix, placeholder=None, camera_instance=None):
    """
    A reusable camera component that handles camera operations
    
    Args:
        key_prefix: Unique prefix for this camera instance
        placeholder: Optional placeholder for the camera feed
        camera_instance: Optional existing camera instance
        
    Returns:
        tuple: (captured_frame, running_status)
    """
    # Set up session state keys with fixed structure
    state_key = f"{key_prefix}_running"
    
    # Initialize camera state
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    
    # Use existing camera or create a new one
    if camera_instance is None:
        if 'camera' not in st.session_state:
            st.session_state.camera = Camera()
        camera_instance = st.session_state.camera
    
    # Camera controls - using fixed keys
    col1, col2 = st.columns(2)
    with col1:
        start_key = f"{key_prefix}_start"
        start_camera = st.button("Start Camera", key=start_key)
    with col2:
        stop_key = f"{key_prefix}_stop"
        stop_camera = st.button("Stop Camera", key=stop_key)
    
    # Use provided placeholder or create a new one
    if placeholder is None:
        preview_placeholder = st.empty()
    else:
        preview_placeholder = placeholder
    
    # Capture button with fixed key
    capture_key = f"{key_prefix}_capture"
    capture_btn = st.button("Capture Image", key=capture_key)
    
    # Handle camera start/stop
    if start_camera:
        st.session_state[state_key] = True
        camera_instance.start()
    
    if stop_camera:
        st.session_state[state_key] = False
        camera_instance.stop()
    
    # Capture logic
    captured_frame = None
    
    # Display camera feed (single frame)
    if st.session_state[state_key]:
        frame = camera_instance.get_frame()
        if frame is not None:
            # Convert color for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(frame_rgb, channels="RGB", caption="Camera Preview")
            
            # Handle capture
            if capture_btn:
                captured_frame = frame.copy()
                st.session_state[state_key] = False
                camera_instance.stop()
    
    # Return the captured frame and running status
    return captured_frame, st.session_state[state_key] 