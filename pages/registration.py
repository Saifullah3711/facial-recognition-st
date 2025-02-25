import streamlit as st
import cv2
import numpy as np
from utils.face_processor import FaceProcessor
from utils.database import Database
from utils.camera import Camera
from utils.camera_component import camera_capture
import time
from PIL import Image
import io
import base64
import uuid
from utils.camera_handlers import registration_camera, edit_camera

def show():
    st.title("User Registration")
    
    # Initialize face processor and database
    if 'face_processor' not in st.session_state:
        st.session_state.face_processor = FaceProcessor()
    
    if 'database' not in st.session_state:
        st.session_state.database = Database()
    
    # Add tabs for registration and user list
    tab1, tab2 = st.tabs(["Add User", "All Users"])
    
    with tab1:
        add_user_page()
    
    with tab2:
        all_users_page()

def add_user_page():
    st.header("Add New User")
    
    # Choose image source
    source = st.radio("Select Image Source:", ("Take Picture", "Upload Image"))
    
    face_image = None
    face_embedding = None
    image_placeholder = st.empty()
    face_exists = False
    existing_user = None
    
    if source == "Take Picture":
        # Use the live camera component
        st.write("Please look directly at the camera and ensure good lighting")
        captured_frame = registration_camera()
        
        # Process captured frame if any
        if captured_frame is not None:
            with st.spinner("Processing face..."):
                # Process captured image
                embedding, face_img, status, message = st.session_state.face_processor.get_face_embedding(captured_frame)
                
                if status:
                    st.success("✅ Face detected successfully!")
                    face_embedding = embedding
                    face_image = face_img
                    
                    # Display the captured face
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(face_rgb, caption="Detected Face")
                    
                    # Check if face exists in database IMMEDIATELY
                    with st.spinner("Checking if user already exists..."):
                        face_exists, existing_user = st.session_state.database.check_face_exists(
                            face_embedding, 
                            similarity_threshold=0.6
                        )
                else:
                    st.error(f"❌ {message}")
                    st.info("Please try again with better lighting and a clear view of your face")
    
    elif source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read the image
            image_bytes = uploaded_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the image
            embedding, face_img, status, message = st.session_state.face_processor.get_face_embedding(img)
            
            if status:
                st.success("Face detected successfully!")
                face_embedding = embedding
                face_image = face_img
                
                # Display the detected face
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                image_placeholder.image(face_rgb, caption="Detected Face")
                
                # Check if face exists in database IMMEDIATELY
                with st.spinner("Checking if user already exists..."):
                    face_exists, existing_user = st.session_state.database.check_face_exists(
                        face_embedding, 
                        similarity_threshold=0.6
                    )
            else:
                st.error(message)
    
    # Show warning if face exists
    if face_exists and existing_user:
        st.error(f"⚠️ This face already exists in the database!")
        st.warning(f"User: {existing_user['name']}")
        st.warning(f"Similarity score: {existing_user['similarity']:.2f}")
        st.info("Please use a different face or update the existing user.")
        return  # Stop here if face exists
    
    # Only show the registration form if face detected AND doesn't exist in database
    if face_embedding is not None and face_image is not None and not face_exists:
        with st.form("registration_form"):
            st.subheader("User Information")
            
            # Form fields
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=1, max_value=120, value=25)
            id_card_number = st.text_input("ID Card Number")
            nationality = st.text_input("Nationality")
            profession = st.text_input("Profession")
            
            # Submit button
            submit_button = st.form_submit_button("Register User")
            
            if submit_button:
                if not name or not id_card_number:
                    st.error("Name and ID Card Number are required")
                else:
                    # Convert face image to base64 for storage
                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    buff = io.BytesIO()
                    pil_img.save(buff, format="JPEG")
                    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
                    
                    # Add user to database
                    success, message, user_id = st.session_state.database.add_user(
                        name=name,
                        age=int(age),
                        id_card_number=id_card_number,
                        nationality=nationality,
                        profession=profession,
                        face_embedding=face_embedding,
                        image_base64=img_str
                    )
                    
                    if success:
                        st.success(f"{message} (User ID: {user_id})")
                    else:
                        st.error(message)

def all_users_page():
    st.header("All Registered Users")
    
    # Get all users from database
    users = st.session_state.database.get_all_users()
    
    if not users:
        st.info("No users registered yet.")
        return
    
    # Handle delete confirmation if it's in session state
    if "delete_user_id" in st.session_state and st.session_state.delete_user_id:
        delete_user_modal(st.session_state.delete_user_id, st.session_state.delete_user_name)
        return  # Exit after showing delete modal to prevent infinite loop
    
    # Handle edit user if it's in session state
    if "edit_user_id" in st.session_state and st.session_state.edit_user_id:
        edit_user_modal(st.session_state.edit_user_id)
        return  # Exit after showing edit modal to prevent infinite loop
    
    # Display users in a grid
    cols_per_row = 3
    
    for i in range(0, len(users), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            if i + j < len(users):
                user = users[i + j]
                
                with cols[j]:
                    st.subheader(user["name"])
                    
                    # Display user image if available
                    if "image_base64" in user and user["image_base64"]:
                        image_bytes = base64.b64decode(user["image_base64"])
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, width=150)
                    
                    # Display user info
                    st.write(f"ID: {user['id_card_number']}")
                    st.write(f"Age: {user['age']}")
                    st.write(f"Nationality: {user['nationality']}")
                    st.write(f"Profession: {user['profession']}")
                    
                    # Edit and delete buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"Edit", key=f"edit_{user['_id']}"):
                            st.session_state.edit_user_id = str(user["_id"])
                            st.rerun()
                    
                    with col2:
                        if st.button(f"Delete", key=f"delete_{user['_id']}"):
                            # Confirm deletion
                            st.session_state.delete_user_id = str(user["_id"])
                            st.session_state.delete_user_name = user["name"]
                            st.rerun()

def edit_user_modal(user_id):
    st.subheader("Edit User")
    
    # Get user data
    user = st.session_state.database.get_user(user_id)
    
    if not user:
        st.error("User not found")
        if st.button("Close"):
            st.session_state.edit_user_id = None
            st.rerun()
        return
    
    # Display current image
    if "image_base64" in user and user["image_base64"]:
        image_bytes = base64.b64decode(user["image_base64"])
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, width=200, caption="Current Image")
    
    # Option to update the image
    update_image = st.checkbox("Update Image")
    
    new_face_embedding = None
    new_image_base64 = None
    
    if update_image:
        # Choose image source
        source = st.radio("Select New Image Source:", ("Take Picture", "Upload Image"), key="edit_image_source")
        
        if source == "Take Picture":
            # Use the new live camera component
            captured_frame = edit_camera(user_id)
            
            # Process captured frame if any
            if captured_frame is not None:
                # Process captured image
                embedding, face_img, status, message = st.session_state.face_processor.get_face_embedding(captured_frame)
                
                if status:
                    st.success("Face detected successfully!")
                    new_face_embedding = embedding
                    
                    # Convert to base64
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    buff = io.BytesIO()
                    pil_img.save(buff, format="JPEG")
                    new_image_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
                    
                    # Display new face
                    st.image(face_rgb, width=200, caption="New Detected Face")
                else:
                    st.error(message)
        
        elif source == "Upload Image":
            uploaded_file = st.file_uploader("Choose a new image file", type=["jpg", "jpeg", "png"], key="edit_image_upload")
            
            if uploaded_file is not None:
                # Read the image
                image_bytes = uploaded_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process the image
                embedding, face_img, status, message = st.session_state.face_processor.get_face_embedding(img)
                
                if status:
                    st.success("Face detected successfully!")
                    new_face_embedding = embedding
                    
                    # Convert to base64
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    buff = io.BytesIO()
                    pil_img.save(buff, format="JPEG")
                    new_image_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
                    
                    # Display new face
                    st.image(face_rgb, width=200, caption="New Detected Face")
                else:
                    st.error(message)
    
    # User information form
    with st.form("edit_user_form"):
        name = st.text_input("Full Name", value=user["name"])
        age = st.number_input("Age", min_value=1, max_value=120, value=user["age"])
        id_card_number = st.text_input("ID Card Number", value=user["id_card_number"])
        nationality = st.text_input("Nationality", value=user["nationality"])
        profession = st.text_input("Profession", value=user["profession"])
        
        # Submit and cancel buttons
        col1, col2 = st.columns(2)
        
        with col1:
            submit = st.form_submit_button("Update")
        
        with col2:
            cancel = st.form_submit_button("Cancel")
    
    if submit:
        # Update user in database
        success, message = st.session_state.database.update_user(
            user_id=user_id,
            name=name,
            age=int(age),
            id_card_number=id_card_number,
            nationality=nationality,
            profession=profession,
            face_embedding=new_face_embedding,
            image_base64=new_image_base64
        )
        
        if success:
            st.success(message)
            st.session_state.edit_user_id = None
            st.rerun()
        else:
            st.error(message)
    
    if cancel:
        st.session_state.edit_user_id = None
        st.rerun()

def delete_user_modal(user_id, user_name):
    st.subheader(f"Delete User: {user_name}")
    st.warning("Are you sure you want to delete this user? This action cannot be undone.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Yes, Delete", key="confirm_delete_btn"):
            with st.spinner("Deleting user..."):
                # Call the delete method
                success, message = st.session_state.database.delete_user(user_id)
                
                if success:
                    # Clear the session state
                    if "delete_user_id" in st.session_state:
                        del st.session_state.delete_user_id
                    
                    if "delete_user_name" in st.session_state:
                        del st.session_state.delete_user_name
                    
                    # Show success message
                    st.success(message)
                    
                    # Use a different approach to reload the page
                    st.rerun()
                else:
                    st.error(message)
    
    with col2:
        if st.button("Cancel", key="cancel_delete_btn"):
            # Clear the session state properly
            if "delete_user_id" in st.session_state:
                del st.session_state.delete_user_id
            
            if "delete_user_name" in st.session_state:
                del st.session_state.delete_user_name
            
            # Rerun the application
            st.rerun() 