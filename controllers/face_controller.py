from flask import Blueprint, jsonify, request
from flask_cors import CORS  # Import CORS
from models.face_model import register_patient, search_patient
from util import convert_base64_to_embedding, normalize_embedding
import logging  # Import logging 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the blueprint
face_blueprint = Blueprint('face', __name__)


@face_blueprint.route('/register_patient', methods=['POST'])
def register_patient_route():
    # Check if request has JSON content
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request must contain JSON data.'}), 400

    # Get mrn and base64_image from JSON body
    data = request.get_json()
    mrn = data.get('mrn', '').strip()
    base64_image = data.get('base64_image', '').strip()
    
    # Validate inputs
    if not mrn:
        return jsonify({'status': 'error', 'message': 'MRN is required.'}), 400
    if not base64_image:
        return jsonify({'status': 'error', 'message': 'Base64 image data is required.'}), 400
    
    # Convert base64 image to embedding vector
    result, status_code = convert_base64_to_embedding(base64_image, is_search=False)
    if status_code != 200 or 'embedding' not in result:
        return jsonify({'status': 'error', 'message': result['status']}), status_code
    
    embedding_vector = result['embedding']
    
    # Call model function with the converted embedding
    result, status_code = register_patient(mrn, embedding_vector)
    return jsonify(result), status_code

@face_blueprint.route('/search_patient', methods=['POST'])
def search_patient_route():
    # Check if request has JSON content
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request must contain JSON data.'}), 400

    # Get base64_image from JSON body
    data = request.get_json()
    is_search = data.get('is_search')
    base64_image = data.get('base64_image', '').strip()
    
    # Validate input
    if not base64_image:
        return jsonify({'status': 'error', 'message': 'Base64 image data is required.'}), 400
    
    # Convert base64 image to embedding vector
    result, status_code = convert_base64_to_embedding(base64_image, is_search)
    if status_code != 200 or 'embedding' not in result:
        return jsonify({'status': 'error', 'message': result['status']}), status_code
    
    embedding_vector = result['embedding']
    
    # Call model function with the embedding
    result, status_code = search_patient(embedding_vector, is_search)
    return jsonify(result), status_code