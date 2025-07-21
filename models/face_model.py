import json
import logging
import numpy as np
from psycopg2 import Error
from util import  normalize_embedding, get_db_connection_ai#, db_pool_ai, release_db_connection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_patient(mrn, embedding_vector):
    #logger.info("Starting register_patient model function")
    
    # Validate inputs
    if not mrn:
        return {'status': 'error', 'message': 'Please provide an MRN.'}, 400
    if not embedding_vector:
        return {'status': 'error', 'message': 'Please provide a valid image.'}, 400
    
    # Validate embedding vector
    if not isinstance(embedding_vector, list) or len(embedding_vector) != 128:
        return {
            'status': 'error',
            'message': 'Invalid embedding vector format. Must be a list of 128 floats.'
        }, 400

    # Normalize the embedding
    normalized_embedding = normalize_embedding(embedding_vector)
    #logger.info(f"Normalized embedding: {normalized_embedding[:5]}...")

    conn = get_db_connection_ai()
    if conn is None:
        return {'status': 'error', 'message': 'Database connection failed.'}, 500
    
    try:
        cur = conn.cursor()
        # Check if MRN already exists
        cur.execute("SELECT mrn FROM face_embeddings WHERE mrn = %s", (mrn,))
        if cur.fetchone():
            #logger.info(f"Register - MRN {mrn} already exists in the database")
            return {
                'status': 'error',
                'message': f'MRN {mrn} is already registered. Please use a unique MRN.'
            }, 409

        # Check for similar face embeddings
        cur.execute(
            """
            SELECT mrn, embedding_vector <-> %s::vector AS distance 
            FROM face_embeddings 
            WHERE embedding_vector IS NOT NULL 
            AND embedding_vector <-> %s::vector < 0.4 
            ORDER BY distance LIMIT 1
            """,
            (str(normalized_embedding), str(normalized_embedding))
        )
        rows = cur.fetchall()
        if rows:
            closest_row = rows[0]
            if closest_row[1] < 0.35:
                return {
                    'status': 'error',
                    'message': f'Face is already registered under MRN: {closest_row[0]}'
                }, 409

        # Insert new embedding
        cur.execute(
            """
            INSERT INTO face_embeddings (mrn, embedding_vector) 
            VALUES (%s, %s::vector)
            """,
            (mrn, str(normalized_embedding))
        )
        conn.commit()
        #logger.info(f"Register - New patient registered with MRN: {mrn}")
        return {
            'status': 'success',
            'message': f'Patient with MRN {mrn} registered successfully!'
        }, 201
    except Error as e:
        logger.error(f"Database error during registration: {e}")
        return {
            'status': 'error',
            'message': 'Registration failed due to database error.'
        }, 500
    '''finally:
        if 'cur' in locals():
            cur.close()
        release_db_connection(conn)'''

def search_patient(embedding_vector, is_search):
    #logger.info("Starting search_patient model function")
    
    # Validate input
    if not embedding_vector:
        return {
            'status': 'error',
            'message': 'Please provide a valid image.'
        }, 400
    
    # Validate embedding vector
    if not isinstance(embedding_vector, list) or len(embedding_vector) != 128:
        return {
            'status': 'error',
            'message': 'Invalid embedding vector format. Must be a list of 128 floats.'
        }, 400

    # Normalize the embedding
    normalized_embedding = normalize_embedding(embedding_vector)

    conn = get_db_connection_ai()
    if conn is None:
        return {
            'status': 'error',
            'message': 'Database connection failed.'
        }, 500
    
    try:
        cur = conn.cursor()
        # Check for similar face embeddings
        cur.execute(
            """
            SELECT mrn, embedding_vector <-> %s::vector AS distance 
            FROM face_embeddings 
            WHERE embedding_vector IS NOT NULL 
            AND embedding_vector <-> %s::vector < 0.4 
            ORDER BY distance LIMIT 1
            """,
            (str(normalized_embedding), str(normalized_embedding))
        )
        rows = cur.fetchall()
        if rows:
            closest_row = rows[0]
            if closest_row[1] < 0.35:
                #logger.info(f"Match found: MRN {closest_row[0]}")
                return {
                    'status': 'success',
                    'data': {'mrn': closest_row[0]}
                }, 200
        #logger.info("No matching patient found")
        if is_search is True:
            return {
                'status': 'error',
                'message': 'Unknown Patient. Please Register the Patient'
            }, 409
        else:
            return {
                'status': 'error',
                'message': 'Unknown Patient. Please Register the Patient'
            }, 200
    except Error as e:
        logger.error(f"Database error during search: {e}")
        return {
            'status': 'error',
            'message': 'Search failed due to database error.'
        }, 500
    '''finally:
        if 'cur' in locals():
            cur.close()
        release_db_connection(conn)'''