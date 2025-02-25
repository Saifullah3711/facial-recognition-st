import os
import pymongo
from pymongo import errors as pymongo_errors
from datetime import datetime
import numpy as np
from bson.binary import Binary
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        # Get MongoDB URI from environment variables
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError("MongoDB URI not found in environment variables")
        
        try:
            # Connect to MongoDB
            logger.info("Connecting to MongoDB Atlas...")
            self.client = pymongo.MongoClient(mongodb_uri)
            
            # Test the connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
            
            # Set up database and collections
            self.db = self.client["face_verification"]
            self.users_collection = self.db["users"]
            self.logs_collection = self.db["logs"]
            
            # Create indexes if they don't exist
            self._initialize_collections()
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise
    
    def _initialize_collections(self):
        """
        Initialize collections and create necessary indexes
        """
        try:
            # Get list of existing collections
            existing_collections = self.db.list_collection_names()
            
            # Create collections if they don't exist
            if "users" not in existing_collections:
                logger.info("Creating 'users' collection...")
                try:
                    self.db.create_collection("users")
                except pymongo.errors.CollectionInvalid:
                    logger.info("Collection 'users' already exists")
            else:
                logger.info("Collection 'users' already exists")
            
            if "logs" not in existing_collections:
                logger.info("Creating 'logs' collection...")
                try:
                    self.db.create_collection("logs")
                except pymongo.errors.CollectionInvalid:
                    logger.info("Collection 'logs' already exists")
            else:
                logger.info("Collection 'logs' already exists")
            
            # Create indexes
            logger.info("Creating indexes...")
            
            # Users collection indexes
            existing_indexes = [idx["name"] for idx in self.users_collection.list_indexes()]
            if "id_card_number_1" not in existing_indexes:
                self.users_collection.create_index("id_card_number", unique=True)
                logger.info("Created index on id_card_number")
            else:
                logger.info("Index on id_card_number already exists")
            
            # Logs collection indexes
            existing_indexes = [idx["name"] for idx in self.logs_collection.list_indexes()]
            if "timestamp_1" not in existing_indexes:
                self.logs_collection.create_index("timestamp")
                logger.info("Created index on timestamp")
            else:
                logger.info("Index on timestamp already exists")
            
            logger.info("Database initialization complete")
            
        except pymongo.errors.CollectionInvalid as e:
            # This is fine - collection already exists
            logger.info(f"Collection initialization note: {str(e)}")
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            # Don't raise the exception, just log it
            # This allows the application to continue even if there's an issue with collection creation
    
    def add_user(self, name, age, id_card_number, nationality, profession, face_embedding, image_base64):
        """
        Add a new user to the database
        
        Args:
            name: User's name
            age: User's age
            id_card_number: ID card number
            nationality: User's nationality
            profession: User's profession
            face_embedding: Numpy array of face embedding
            image_base64: Base64 encoded image
            
        Returns:
            tuple: (success, message, user_id)
        """
        try:
            # Check if user with same ID already exists
            existing_user = self.users_collection.find_one({"id_card_number": id_card_number})
            if existing_user:
                return False, f"User with ID {id_card_number} already exists", None
            
            # Serialize the numpy array for MongoDB
            embedding_binary = Binary(pickle.dumps(face_embedding, protocol=2), subtype=128)
            
            # Create user document
            user = {
                "name": name,
                "age": age,
                "id_card_number": id_card_number,
                "nationality": nationality,
                "profession": profession,
                "face_embedding": embedding_binary,
                "image_base64": image_base64,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Insert user
            result = self.users_collection.insert_one(user)
            logger.info(f"Added new user: {name} with ID: {id_card_number}")
            
            return True, "User added successfully", str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error adding user: {str(e)}")
            return False, f"Error adding user: {str(e)}", None
    
    def get_all_users(self):
        """
        Get all users from the database
        
        Returns:
            list: List of user documents
        """
        users = list(self.users_collection.find({}, {
            "face_embedding": 0  # Exclude face embedding for listing
        }))
        
        return users
    
    def get_user(self, user_id):
        """
        Get a user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            dict: User document
        """
        from bson.objectid import ObjectId
        
        user = self.users_collection.find_one({"_id": ObjectId(user_id)})
        
        # Deserialize the face embedding
        if user and "face_embedding" in user:
            user["face_embedding"] = pickle.loads(user["face_embedding"])
            
        return user
    
    def update_user(self, user_id, name, age, id_card_number, nationality, profession, face_embedding=None, image_base64=None):
        """
        Update a user in the database
        
        Args:
            user_id: User ID
            name: User's name
            age: User's age
            id_card_number: ID card number
            nationality: User's nationality
            profession: User's profession
            face_embedding: Numpy array of face embedding (optional)
            image_base64: Base64 encoded image (optional)
            
        Returns:
            tuple: (success, message)
        """
        try:
            from bson.objectid import ObjectId
            
            # Check if another user with same ID exists
            existing_user = self.users_collection.find_one({
                "id_card_number": id_card_number,
                "_id": {"$ne": ObjectId(user_id)}
            })
            
            if existing_user:
                return False, f"Another user with ID {id_card_number} already exists"
            
            # Create update document
            update_doc = {
                "name": name,
                "age": age,
                "id_card_number": id_card_number,
                "nationality": nationality,
                "profession": profession,
                "updated_at": datetime.now()
            }
            
            # Add face embedding if provided
            if face_embedding is not None:
                update_doc["face_embedding"] = Binary(pickle.dumps(face_embedding, protocol=2), subtype=128)
            
            # Add image if provided
            if image_base64 is not None:
                update_doc["image_base64"] = image_base64
            
            # Update user
            result = self.users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_doc}
            )
            
            if result.matched_count == 0:
                return False, "User not found"
                
            return True, "User updated successfully"
            
        except Exception as e:
            return False, f"Error updating user: {str(e)}"
    
    def delete_user(self, user_id):
        """
        Delete a user from the database
        
        Args:
            user_id: ID of the user to delete
            
        Returns:
            tuple: (success, message)
        """
        try:
            from bson.objectid import ObjectId
            
            # Check if user exists
            user = self.users_collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                return False, f"User with ID {user_id} not found"
            
            # Delete user
            result = self.users_collection.delete_one({"_id": ObjectId(user_id)})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted user with ID: {user_id}")
                return True, "User deleted successfully"
            else:
                return False, "User not deleted. Please try again."
            
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False, f"Error deleting user: {str(e)}"
    
    def get_all_embeddings(self):
        """
        Get all face embeddings for verification
        
        Returns:
            list: List of tuples (user_id, user_data, embedding)
        """
        users = list(self.users_collection.find({}))
        
        embeddings = []
        for user in users:
            # Deserialize the face embedding
            if "face_embedding" in user:
                embedding = pickle.loads(user["face_embedding"])
                user_data = {
                    "name": user["name"],
                    "id_card_number": user["id_card_number"],
                    "nationality": user["nationality"],
                    "profession": user["profession"]
                }
                embeddings.append((str(user["_id"]), user_data, embedding))
                
        return embeddings
    
    def add_log(self, person_id, person_name, recognition_status, confidence_score, image_base64):
        """
        Add a log entry
        
        Args:
            person_id: Person ID if recognized, None otherwise
            person_name: Person name if recognized, None otherwise
            recognition_status: 'recognized' or 'unknown'
            confidence_score: Confidence score of recognition
            image_base64: Base64 encoded small image
            
        Returns:
            tuple: (success, message)
        """
        try:
            log = {
                "timestamp": datetime.now(),
                "person_id": person_id,
                "person_name": person_name,
                "recognition_status": recognition_status,
                "confidence_score": confidence_score,
                "image_base64": image_base64
            }
            
            self.logs_collection.insert_one(log)
            
            return True, "Log added successfully"
            
        except Exception as e:
            return False, f"Error adding log: {str(e)}"
    
    def get_logs(self, hours=None):
        """
        Get logs, optionally filtered by hours
        
        Args:
            hours: Number of hours to look back, None for all logs
            
        Returns:
            list: List of log documents
        """
        try:
            query = {}
            
            if hours:
                from datetime import timedelta
                query["timestamp"] = {
                    "$gte": datetime.now() - timedelta(hours=hours)
                }
            
            # Important: We explicitly convert to list here
            logs = list(self.logs_collection.find(query).sort("timestamp", -1))
            
            # Ensure we're returning a list and not a cursor
            if not isinstance(logs, list):
                logs = list(logs)
            
            return logs
        except Exception as e:
            # Return empty list on error
            return []
    
    def add_recognition_log(self, recognition_status, person_id, person_name, confidence_score, face_image=None):
        """
        Add a recognition log entry to the database
        
        Args:
            recognition_status: 'recognized' or 'unknown'
            person_id: ID of the recognized person (or None)
            person_name: Name of the recognized person (or None)
            confidence_score: Confidence score of recognition
            face_image: Base64 encoded image of the detected face
            
        Returns:
            success: True if operation was successful
            message: Success or error message
        """
        try:
            # Create logs collection if it doesn't exist
            if 'recognition_logs' not in self.db.list_collection_names():
                self.db.create_collection('recognition_logs')
            
            # Create log entry
            log_entry = {
                'timestamp': datetime.now(),
                'recognition_status': recognition_status,
                'person_id': person_id,
                'person_name': person_name,
                'confidence_score': confidence_score,
                'face_image': face_image
            }
            
            # Insert log entry
            self.db.recognition_logs.insert_one(log_entry)
            
            return True, "Log added successfully"
        except Exception as e:
            return False, f"Error adding log: {str(e)}"
    
    def check_face_exists(self, face_embedding, similarity_threshold=0.5):
        """
        Check if a face with similar embedding already exists in the database
        
        Args:
            face_embedding: Numpy array of face embedding to check
            similarity_threshold: Threshold for face similarity (0-1), lower means stricter matching
            
        Returns:
            tuple: (exists, user_info) - whether the face exists and user info if found
        """
        try:
            # Get all users
            users = list(self.users_collection.find({}))
            
            for user in users:
                if "face_embedding" in user:
                    # Deserialize face embedding
                    existing_embedding = pickle.loads(user["face_embedding"])
                    
                    # Calculate similarity (using cosine similarity)
                    similarity = np.dot(existing_embedding, face_embedding) / (
                        np.linalg.norm(existing_embedding) * np.linalg.norm(face_embedding)
                    )
                    
                    # If similarity is above threshold, face exists
                    if similarity > similarity_threshold:
                        return True, {
                            "id": str(user["_id"]),
                            "name": user["name"],
                            "similarity": similarity
                        }
            
            # No similar face found
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking face existence: {str(e)}")
            return False, None 