import streamlit as st
import cv2
import numpy as np
import io
from utils.camera import Camera

# Global camera instance
if 'camera' not in st.session_state:
    st.session_state.camera = Camera()

def native_camera_capture(key_prefix):
    """
    Camera component using Streamlit's built-in st.camera_input
    
    Args:
        key_prefix: Unique prefix for this camera instance
        
    Returns:
        captured_frame: OpenCV-compatible frame if captured, None otherwise
    """
    # Show camera input with a unique key
    img_file = st.camera_input("Take a picture", key=f"{key_prefix}_camera")
    
    # Process the captured image if available
    if img_file is not None:
        # Convert to OpenCV format
        bytes_data = img_file.getvalue()
        np_array = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        return frame
    
    return None

# Define the public functions that are imported
def registration_camera():
    """Camera component for registration using native Streamlit"""
    return native_camera_capture("reg")

def edit_camera(user_id):
    """Camera component for editing using native Streamlit"""
    return native_camera_capture(f"edit_{user_id}")

def realtime_recognition_camera():
    """Camera component for real-time recognition using native Streamlit"""
    # Show camera input with a unique key
    st.subheader("Take a snapshot for recognition")
    img_file = st.camera_input("Capture for recognition", key="realtime_camera")
    
    # Process the captured image if available
    if img_file is not None:
        # Convert to OpenCV format
        bytes_data = img_file.getvalue()
        np_array = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        return frame
    
    return None 