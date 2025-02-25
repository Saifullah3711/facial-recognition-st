import streamlit as st
import cv2
import time
import numpy as np
from datetime import datetime
from utils.face_processor import FaceProcessor
from utils.database import Database
from utils.camera import Camera
from utils.camera_handlers import realtime_recognition_camera
from PIL import Image
import io
import base64
from components.logs_viewer import show_logs_viewer

def add_recognition_to_logs(database, recognition_status, person_id, person_name, confidence_score, face_image):
    """Helper function to add recognition logs"""
    try:
        # Save to database using the existing add_log method
        database.add_log(
            person_id=person_id,
            person_name=person_name,
            recognition_status=recognition_status,
            confidence_score=confidence_score,
            image_base64=face_image
        )
        return True
    except Exception as e:
        st.error(f"Error logging recognition: {str(e)}")
        return False

def show():
    st.title("Real-time Face Recognition")
    
    # Initialize face processor and database if not already done
    if 'face_processor' not in st.session_state:
        st.session_state.face_processor = FaceProcessor()
        
    if 'database' not in st.session_state:
        st.session_state.database = Database()
    
    # Initialize logs list if not exists
    if 'recognition_logs' not in st.session_state:
        st.session_state.recognition_logs = []
    
    # Restore the tabs
    tab1, tab2 = st.tabs(["Face Recognition", "Activity Logs"])
    
    with tab1:
        live_detection_page()
    
    with tab2:
        try:
            show_logs_viewer(st.session_state.database)
        except Exception as e:
            st.error(f"Error displaying logs: {str(e)}")
            st.info("Please try again later.")

def live_detection_page():
    st.header("Face Recognition")
    
    # Clear logs button
    if st.button("Clear Current Logs", key="clear_logs_btn"):
        st.session_state.recognition_logs = []
        st.success("Logs cleared")
    
    # Camera view and logs side by side
    col1, col2 = st.columns([2, 1])
    
    # Camera feed placeholder
    with col1:
        # Get all embeddings from database
        embeddings = st.session_state.database.get_all_embeddings()
        
        if not embeddings:
            st.warning("No registered users found in the database.")
            st.info("Please register at least one user before using face recognition.")
        else:
            st.write("📸 Capture an image to identify the person")
            
            # Use the native camera component
            frame = realtime_recognition_camera()
            
            if frame is not None:
                # Process frame for face detection and recognition
                processed_frame, matches = st.session_state.face_processor.detect_face_realtime(
                    frame, embeddings
                )
                
                # Display the processed frame with recognition results
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(processed_rgb, channels="RGB", caption="Recognition Result")
                
                # Handle matches
                for match_data in matches:
                    # Prepare log entry
                    if match_data["recognized"]:
                        recognition_status = "recognized"
                        person_id = match_data["user_id"]
                        person_name = match_data["match"]
                        confidence_score = match_data["confidence"]
                        details = f"Recognized as {match_data['match']} with {match_data['confidence']:.2f} confidence"
                    else:
                        recognition_status = "unknown"
                        person_id = None
                        person_name = None
                        confidence_score = 0
                        details = "Unknown person detected"
                    
                    # Display the result
                    result_box = st.container()
                    with result_box:
                        if match_data["recognized"]:
                            st.success(f"✅ {details}")
                        else:
                            st.warning(f"⚠️ {details}")
                    
                    # Add to session logs
                    log_entry = {
                        "timestamp": datetime.now(),
                        "recognition_status": recognition_status,
                        "person_id": person_id,
                        "person_name": person_name,
                        "confidence_score": confidence_score,
                        "details": details,
                        "image": match_data["face_image"]
                    }
                    
                    # Save to database
                    try:
                        # Extract name from match data if it's a dictionary
                        display_name = match_data["match"]
                        if isinstance(display_name, dict) and "name" in display_name:
                            display_name = display_name["name"]
                        
                        # Save to database using the existing add_log method
                        st.session_state.database.add_log(
                            person_id=person_id,
                            person_name=display_name,  # Use the extracted name
                            recognition_status=recognition_status,
                            confidence_score=confidence_score,
                            image_base64=match_data["face_image"]
                        )
                    except Exception as e:
                        st.error(f"Error saving log: {str(e)}")
                    
                    # Add to session logs
                    st.session_state.recognition_logs.insert(0, log_entry)
                    
                    # Keep only last 50 logs in session
                    if len(st.session_state.recognition_logs) > 50:
                        st.session_state.recognition_logs = st.session_state.recognition_logs[:50]
    
    # Logs display
    with col2:
        st.subheader("Recent Activity")
        display_logs(st.session_state.recognition_logs)

def display_logs(logs, max_logs=10):
    """Display recent logs in the sidebar"""
    placeholder = st.empty()
    
    if not logs:
        placeholder.info("No recent activity")
        return
    
    logs_md = ""
    
    for i, log in enumerate(logs[:max_logs]):
        time_str = log["timestamp"].strftime("%H:%M:%S")
        
        # Get and format person name properly
        person_name = log["person_name"]
        person_details = {}
        
        # Handle dictionary person names and extract details
        if isinstance(person_name, dict):
            # Save the details for display
            person_details = person_name.copy()
            
            # Extract just the name for the header
            if "name" in person_name:
                person_name = person_name["name"]
            else:
                # Try to get the first value from the dictionary
                try:
                    person_name = next(iter(person_name.values()))
                except:
                    person_name = str(person_name)
        
        # Format the log entry
        if log["recognition_status"] == "recognized":
            # Add the person name as header
            logs_md += f"**✅ {person_name}** ({time_str})\n\n"
            
            # Add confidence score
            logs_md += f"**Confidence:** {log['confidence_score']:.2f}\n\n"
            
            # Add additional details if available
            if person_details:
                logs_md += "**Person Details:**\n\n"
                
                # Display each detail except name (already shown in header)
                for key, value in person_details.items():
                    if key != "name":  # Skip name as it's already in the header
                        # Format the key with title case
                        formatted_key = key.replace('_', ' ').title()
                        logs_md += f"- {formatted_key}: {value}\n"
                logs_md += "\n"
        else:
            logs_md += f"**⚠️ Unknown Person** ({time_str})\n\n"
        
        if log["image"]:
            # Display the face image using HTML
            img_html = f'<img src="data:image/jpeg;base64,{log["image"]}" width="100">'
            logs_md += f"{img_html}\n\n"
        
        logs_md += "---\n\n"
    
    placeholder.markdown(logs_md, unsafe_allow_html=True) 