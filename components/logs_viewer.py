import streamlit as st
import pandas as pd
import base64
import io
from datetime import datetime
from PIL import Image
import altair as alt

def show_logs_viewer(database):
    """Display recognition logs with statistics"""
    st.header("Activity Logs")
    
    try:
        # Get logs from database
        logs = database.get_logs()
        
        if not logs:
            st.info("No logs available")
            return
        
        # Process logs for display
        clean_logs = []
        recognized_count = 0
        unknown_count = 0
        person_stats = {}
        
        for log in logs:
            # Extract key information safely
            try:
                timestamp = log.get("timestamp", datetime.now())
                status = log.get("recognition_status", "unknown")
                
                # Safely get person name - could be a string or dict
                person_name = log.get("person_name", "Unknown")
                
                # Handle dictionary person names properly
                if isinstance(person_name, dict):
                    if "name" in person_name:
                        # Extract name from dictionary
                        person_name = person_name["name"]
                    else:
                        # Just use the first value in the dictionary
                        try:
                            person_name = next(iter(person_name.values()))
                        except:
                            person_name = str(person_name)
                elif not isinstance(person_name, str):
                    # Convert other non-string types to string
                    person_name = str(person_name)
                
                # Count by status
                if status == "recognized":
                    recognized_count += 1
                    
                    # Count by person for recognized faces
                    if person_name not in person_stats:
                        person_stats[person_name] = 0
                    person_stats[person_name] += 1
                else:
                    unknown_count += 1
                
                # Add to cleaned logs with properly formatted person name
                clean_logs.append({
                    "timestamp": timestamp,
                    "status": status,
                    "person_name": person_name,
                    "confidence": log.get("confidence_score", 0.0),
                    "image": log.get("image_base64", None) or log.get("face_image", None)
                })
            except Exception as e:
                st.warning(f"Skipped a log entry due to error: {str(e)}")
        
        # Display statistics
        st.subheader("Recognition Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Recognition status stats
            st.metric("✅ Recognized", recognized_count)
            st.metric("❌ Unknown", unknown_count)
            
            # Create simple chart using pandas
            status_df = pd.DataFrame({
                "Status": ["Recognized", "Unknown"],
                "Count": [recognized_count, unknown_count]
            })
            
            # Simple bar chart
            st.bar_chart(status_df.set_index("Status"))
        
        with col2:
            # Most recognized people
            st.write("Top recognized people:")
            
            if person_stats:
                # Sort by count
                sorted_people = sorted(person_stats.items(), key=lambda x: x[1], reverse=True)
                top_people = sorted_people[:5]  # Top 5
                
                # Create dataframe
                people_df = pd.DataFrame({
                    "Person": [p[0] for p in top_people],
                    "Count": [p[1] for p in top_people]
                })
                
                # Display as table
                st.dataframe(people_df)
                
                # Simple bar chart
                st.bar_chart(people_df.set_index("Person"))
            else:
                st.info("No recognized people in logs")
        
        # Display individual logs
        st.subheader("Recent Logs")
        for log in clean_logs[:20]:  # Show most recent 20 logs
            # Format timestamp
            time_str = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            # Create expander for each log
            if log["status"] == "recognized":
                expander_title = f"✅ {time_str} - {log['person_name']} ({log['confidence']:.2f})"
            else:
                expander_title = f"❌ {time_str} - Unknown Person"
            
            with st.expander(expander_title):
                # Display in columns
                cols = st.columns([1, 3])
                
                # Display image if available
                with cols[0]:
                    if log["image"]:
                        try:
                            image_bytes = base64.b64decode(log["image"])
                            image = Image.open(io.BytesIO(image_bytes))
                            st.image(image, width=100)
                        except:
                            st.write("Image not available")
                
                # Display details
                with cols[1]:
                    st.write(f"**Time:** {time_str}")
                    st.write(f"**Status:** {log['status']}")
                    
                    if log["status"] == "recognized":
                        st.write(f"**Person:** {log['person_name']}")
                        st.write(f"**Confidence:** {log['confidence']:.4f}")
    
    except Exception as e:
        st.error(f"Error loading logs: {str(e)}")
        import traceback
        st.code(traceback.format_exc()) 