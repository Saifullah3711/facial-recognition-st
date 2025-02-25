import streamlit as st
import os
from dotenv import load_dotenv
import pages.registration as registration
import pages.realtime_test as realtime_test

# Load environment variables
load_dotenv()

# Configure the page layout
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS to hide default navigation and improve UI
hide_streamlit_style = """
<style>
    /* Hide the default Streamlit navigation/menu in sidebar */
    [data-testid="stSidebarNav"] {display: none !important;}
    
    /* Custom styling for the app */
    .main-header {font-size: 2rem; font-weight: bold; margin-bottom: 1rem;}
    .app-description {margin-bottom: 2rem; color: #666;}
    .stRadio > label {font-weight: bold; font-size: 1.2rem;}
    
    /* Make the app more modern */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Create a clean sidebar for navigation
with st.sidebar:
    st.title("Face Recognition")
    
    # Navigation options
    page = st.radio(
        "Navigation",
        ["Home", "Registration", "Face Recognition"]
    )
    
    st.markdown("---")
    st.caption("© 2025 Face Recognition System by Saif")

# Display the selected page
if page == "Home":
    st.markdown("<h1 class='main-header'>Welcome to Face Recognition System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='app-description'>
        This application allows you to register users with facial recognition and verify their identity in real-time.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Key Features")
        st.markdown("""
        - **User Registration** - Register new users with facial data
        - **Real-time Recognition** - Identify registered users in real-time
        - **Activity Logging** - Track all recognition events
        """)
        
        st.markdown("### How to Use")
        st.markdown("""
        1. Start with **Registration** to add users to the system
        2. Capture photos or upload images of users' faces
        3. Switch to **Face Recognition** to identify registered users
        4. View logs to track recognition activity
        """)
    
elif page == "Registration":
    registration.show()
elif page == "Face Recognition":
    realtime_test.show() 