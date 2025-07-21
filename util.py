import os
import yaml
import psycopg2
from psycopg2 import Error
import logging
from psycopg2.pool import ThreadedConnectionPool
import base64
import numpy as np
import face_recognition
from PIL import Image
import io
import time
from psycopg2 import OperationalError, InterfaceError
#from Antispoofing.test import test  # Import the test function from test.py

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load database configuration
try:
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    if config is None:
        raise ValueError("config.yml is empty or invalid")
    DB_CONFIG_AI = config['database']['backend_ai']
except FileNotFoundError:
    logger.error("config.yml not found")
    raise
except KeyError as e:
    logger.error(f"Missing key in config.yml: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading config.yml: {e}")
    raise

# Create a connection pool for Backend_AI
'''try:
    db_pool_ai = ThreadedConnectionPool(1, 20, **DB_CONFIG_AI)
except Exception as e:
    logger.error(f"Error initializing database connection pool: {e}")
    raise

def get_db_connection_ai():
    # Establish a connection to Backend_AI from the pool.
    try:
        return db_pool_ai.getconn()
    except Error as e:
        #logger.error(f"Error connecting to Backend_AI: {e}")
        return None

def release_db_connection(conn):
    # Release the connection back to the pool.
    if conn:
        db_pool_ai.putconn(conn)'''
        
def get_db_connection_ai():
    try:
        conn = psycopg2.connect(**DB_CONFIG_AI)
        return conn
    except psycopg2.Error as e:
        logger.error(f"Direct DB connection error: {e}")
        return None

def normalize_embedding(embedding_vector):
    
    #Normalize the embedding vector to unit length, Returns: normalized embedding as a list
    embedding_array = np.array(embedding_vector)
    norm = np.linalg.norm(embedding_array)
    if norm == 0:
        return embedding_vector  # Avoid division by zero
    normalized = (embedding_array / norm).tolist()
    return normalized

'''def check_anti_spoofing(image_array, model_dir='Antispoofing/resources/anti_spoof_models'):
    
    try:
        label = test(image=image_array, model_dir=model_dir)
        logger.info(f"Anti-spoofing result: label = {label}")
        if label != 1:
            return False, "Spoof detected! Image is not a real face."
        return True, None
    except Exception as e:
        logger.error(f"Error in anti-spoofing check: {e}")
        return False, f"Anti-spoofing check failed: {str(e)}"'''

#def convert_base64_to_embedding(base64_image, antispoofing=False):   #antispoofing: If True, perform anti-spoofing check; if False, skip it
def convert_base64_to_embedding(base64_image, is_search, antispoofing=False):
    if not base64_image:
        return {"status": "No image data provided"}, 400
    try:
        # Decode base64 string to bytes
        image_data = base64.b64decode(base64_image)
        
        # Load image and convert to RGB
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize image to max 500x500 while maintaining aspect ratio
        max_size = 500
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image_array = np.array(image)
        
        '''# Perform anti-spoofing check if enabled
        if antispoofing:
            is_real, error_message = check_anti_spoofing(image_array)
            if not is_real:
                return None, error_message'''
        
        # Detect face locations with no upsampling
        face_locations = face_recognition.face_locations(image_array, number_of_times_to_upsample=0)
        
        if is_search is True:
            if not face_locations:
                return {"status": "No faces detected in the image."}, 409
        else:
            if not face_locations:
                return {"status": "No faces detected"}, 200
        
        embeddings = face_recognition.face_encodings(image_array, known_face_locations=face_locations, num_jitters=0)
        if not embeddings:
            return {"status": "Failed to generate embedding"}, 400
        
        embedding_vector = embeddings[0].tolist()
        if len(embedding_vector) != 128:
            return {"status": "Generated embedding vector is not 128-dimensional"}, 400
        
        normalized_embedding = normalize_embedding(embedding_vector)
        return {"embedding": normalized_embedding, "status": "Success"}, 200
    except base64.binascii.Error:
        return {"status": "Invalid base64 image data"}, 400
    except Exception as e:
        return {"status": f"Failed to process image: {str(e)}"}, 500

