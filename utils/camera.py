import cv2
import threading
import time

class Camera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.is_running = False
        self.lock = threading.Lock()
        self.frame = None
        self.thread = None
        
    def start(self):
        """Start the camera thread if it's not already running"""
        if self.is_running:
            return
            
        # Try different camera indices
        for camera_id in range(5):  # Try camera indices 0-4
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                print(f"Successfully opened camera with ID {camera_id}")
                break
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start thread to read frames
        self.is_running = True
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()
        
        # Give camera time to warm up
        time.sleep(1.0)
    
    def stop(self):
        """Stop the camera thread"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
    
    def _read_frames(self):
        """Thread function that reads frames from the camera"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)  # Short sleep to prevent CPU overuse
    
    def get_frame(self):
        """Get the most recent frame from the camera"""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    
    def capture_image(self):
        """Capture a single image"""
        return self.get_frame()
    
    def __del__(self):
        """Cleanup when object is deleted"""
        self.stop() 