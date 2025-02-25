import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self):
        try:
            # Try to import and initialize InsightFace
            import insightface
            from insightface.app import FaceAnalysis
            logger.info("Using InsightFace for face recognition")
            
            # Initialize the InsightFace model
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            self.use_insightface = True
            
        except ImportError as e:
            logger.warning(f"InsightFace import error: {e}")
            logger.warning("Falling back to OpenCV face detection (less accurate)")
            
            # Fall back to OpenCV's face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.use_insightface = False
    
    def get_face_embedding(self, image):
        """
        Extract face embedding from an image
        
        Args:
            image: Image in BGR format (OpenCV format)
            
        Returns:
            tuple: (face_embedding, face_image, status, message)
                - face_embedding: Numpy array of the face embedding vector or None
                - face_image: Cropped face image or None
                - status: Boolean indicating success or failure
                - message: Success or error message
        """
        if self.use_insightface:
            # Use InsightFace
            # Detect faces
            faces = self.face_app.get(image)
            
            # Check if any face was detected
            if len(faces) == 0:
                return None, None, False, "No face detected in the image"
            
            # If multiple faces detected, take the largest one
            if len(faces) > 1:
                # Sort faces by area (largest first)
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            
            face = faces[0]
            embedding = face.embedding
            
            # Crop the face image
            bbox = face.bbox.astype(int)
            face_image = image[max(0, bbox[1]):min(bbox[3], image.shape[0]), 
                               max(0, bbox[0]):min(bbox[2], image.shape[1])]
            
            return embedding, face_image, True, "Face detected successfully"
        else:
            # Fall back to OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None, None, False, "No face detected in the image"
            
            # Get the largest face
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                
            x, y, w, h = faces[0]
            face_image = image[y:y+h, x:x+w]
            
            # For OpenCV fallback, we'll use a simplified embedding
            # This is not as accurate as InsightFace but allows the app to work
            # We'll use the resized face as a flattened vector
            small_face = cv2.resize(face_image, (50, 50))
            embedding = small_face.flatten() / 255.0  # Normalize to 0-1
            
            return embedding, face_image, True, "Face detected (using OpenCV fallback)"
    
    def detect_face_realtime(self, image, stored_embeddings=None):
        """
        Detect faces in real-time image and match with stored embeddings
        
        Args:
            image: Image in BGR format (OpenCV format)
            stored_embeddings: List of tuples with (user_id, name, embedding)
            
        Returns:
            tuple: (image_with_boxes, matches)
                - image_with_boxes: Image with bounding boxes drawn
                - matches: List of matching user data
        """
        # Make a copy of the image to draw on
        display_image = image.copy()
        matches = []
        
        if self.use_insightface:
            # Use InsightFace
            # Detect faces
            faces = self.face_app.get(image)
            
            # Process each detected face
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                
                # Default to unrecognized (red box)
                color = (0, 0, 255)  # Red
                match = None
                confidence = 0
                
                # Match against stored embeddings if provided
                if stored_embeddings:
                    # Find the best match
                    best_match = None
                    best_score = -1
                    threshold = 0.5  # Cosine similarity threshold
                    
                    for user_id, user_data, stored_embedding in stored_embeddings:
                        # Calculate cosine similarity
                        similarity = np.dot(embedding, stored_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > threshold and similarity > best_score:
                            best_score = similarity
                            best_match = (user_id, user_data, similarity)
                    
                    # If match found
                    if best_match:
                        color = (0, 255, 0)  # Green
                        match = best_match
                        confidence = best_match[2]
                
                # Draw bounding box
                cv2.rectangle(display_image, 
                             (max(0, bbox[0]), max(0, bbox[1])), 
                             (min(bbox[2], display_image.shape[1]), min(bbox[3], display_image.shape[0])), 
                             color, 2)
                
                # Store match data
                face_crop = image[max(0, bbox[1]):min(bbox[3], image.shape[0]), 
                                 max(0, bbox[0]):min(bbox[2], image.shape[1])]
                
                face_data = {
                    "bbox": bbox.tolist(),
                    "match": match[1] if match else None,
                    "user_id": match[0] if match else None,
                    "confidence": float(confidence),
                    "recognized": match is not None,
                    "face_image": self.encode_image_to_base64(face_crop)
                }
                
                matches.append(face_data)
        else:
            # Fall back to OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_crop = image[y:y+h, x:x+w]
                
                # Default to unrecognized (red box)
                color = (0, 0, 255)  # Red
                match = None
                confidence = 0
                
                # Create a simplified embedding
                small_face = cv2.resize(face_crop, (50, 50))
                embedding = small_face.flatten() / 255.0
                
                # Match against stored embeddings if provided
                if stored_embeddings:
                    best_match = None
                    best_score = -1
                    threshold = 0.8  # Higher threshold for OpenCV fallback
                    
                    for user_id, user_data, stored_embedding in stored_embeddings:
                        # For OpenCV fallback, we need to ensure dimensions match
                        if len(embedding) == len(stored_embedding):
                            # Calculate distance (lower is better)
                            distance = np.linalg.norm(embedding - stored_embedding)
                            similarity = 1.0 / (1.0 + distance)  # Convert to similarity
                            
                            if similarity > threshold and similarity > best_score:
                                best_score = similarity
                                best_match = (user_id, user_data, similarity)
                
                    # If match found
                    if best_match:
                        color = (0, 255, 0)  # Green
                        match = best_match
                        confidence = best_match[2]
                
                # Draw bounding box
                cv2.rectangle(display_image, (x, y), (x+w, y+h), color, 2)
                
                # Store match data
                face_data = {
                    "bbox": [x, y, x+w, y+h],
                    "match": match[1] if match else None,
                    "user_id": match[0] if match else None,
                    "confidence": float(confidence),
                    "recognized": match is not None,
                    "face_image": self.encode_image_to_base64(face_crop)
                }
                
                matches.append(face_data)
        
        return display_image, matches
    
    @staticmethod
    def encode_image_to_base64(image):
        """Convert an OpenCV image to base64 string"""
        if image is None or image.size == 0:
            return None
            
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    @staticmethod
    def decode_base64_to_image(base64_string):
        """Convert a base64 string to an OpenCV image"""
        if base64_string is None:
            return None
            
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img 