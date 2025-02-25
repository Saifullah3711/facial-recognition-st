# Face Verification System

A facial verification MVP built using InsightFace, Streamlit, OpenCV, and MongoDB Atlas.

## Features

- User registration with face detection and metadata
- View, edit, and delete registered users
- Real-time face detection and recognition
- Activity logs with filtering and statistics
- MongoDB Atlas integration for cloud storage

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- A MongoDB Atlas account (free tier is sufficient)
- Webcam for real-time recognition

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/Saifullah3711/facial-recognition-st
   
   cd facial-recognition-st
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your MongoDB Atlas connection string:
   ```
   MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>?retryWrites=true&w=majority
   ```

   Replace `<username>`, `<password>`, `<cluster>`, and `<dbname>` with your actual MongoDB Atlas details.

### Windows Installation Notes

If you encounter DLL loading errors with ONNX or InsightFace, try these solutions:

1. **RECOMMENDED SOLUTION**: Downgrade ONNX to version 1.16.1 (this fixes the "DLL load failed while importing onnx_cpp2py_export" error):
   ```
   pip install onnx==1.16.1
   ```

2. Install the Visual C++ Redistributable:
   - Download and install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Restart your system
   - Run the application again


3. If issues persist, the application will automatically fall back to using OpenCV's face detection,
   which is less accurate but more compatible.

Reference: [ONNX GitHub Issue #6267](https://github.com/onnx/onnx/issues/6267)

### Running the Application

Run the Streamlit app:
