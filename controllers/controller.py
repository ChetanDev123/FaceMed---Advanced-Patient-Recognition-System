from flask import Blueprint, jsonify, request
from flask_cors import CORS  # Import CORS
from models.visit_model import train_visit_model
from models.op_model import train_op_model
from models.ip_model import train_ip_model
from models.lab_test_model import get_lab_results
import logging  # Import logging 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the blueprint
visit_blueprint = Blueprint('hati', __name__)

@visit_blueprint.route('/0', methods=['GET'])
def predict_visit():
    """Return visit analytics: busiest days, peak hours, top doctors, top departments, and total visits."""
    top_days, peak_hour_ranges, accuracy, top_doctors, top_departments, total_visits = train_visit_model()

    response_data = {
        "busiest_days": [{"day": day, "peak_hours": peak_hour_ranges[day]} for day in top_days],
        "top_doctors": [{"doctor_name": doctor_name, "visit_count": count} for doctor_name, count in top_doctors.items()],
        "top_departments": [{"department_name": dept_name, "visit_count": count} for dept_name, count in top_departments.items()],
        "total_visits": total_visits
        #"model_accuracy": f"{accuracy * 100:.2f}%"  # Commented out as in original
    }

    return jsonify(response_data)


@visit_blueprint.route('/1', methods=['GET'])
def predict_op():
    """Return OP analytics: busiest days, peak hours, top doctors, top departments, and total OP visits."""
    top_days, peak_hour_ranges, accuracy, top_doctors, top_departments, total_op_visits = train_op_model()

    response_data = {
        "busiest_days": [{"day": day, "peak_hours": peak_hour_ranges[day]} for day in top_days],
        "top_doctors": [{"doctor_name": doctor_name, "visit_count": count} for doctor_name, count in top_doctors.items()],
        "top_departments": [{"department_name": dept_name, "visit_count": count} for dept_name, count in top_departments.items()],
        "total_op_visits": total_op_visits
        #"model_accuracy": f"{accuracy * 100:.2f}%"
    }

    return jsonify(response_data)


@visit_blueprint.route('/2', methods=['GET'])
def predict_ip():
    """Return IP analytics: busiest days, peak hours, top departments, and total IP visits."""
    top_days, peak_hour_ranges, accuracy, top_doctors, top_departments, total_ip_visits = train_ip_model()

    response_data = {
        "busiest_days": [{"day": day, "peak_hours": peak_hour_ranges[day]} for day in top_days],
        "top_doctors": [{"doctor_name": doctor_name, "visit_count": count} for doctor_name, count in top_doctors.items()],
        "top_departments": [{"department_name": dept_name, "visit_count": count} for dept_name, count in top_departments.items()],
        "total_ip_visits": total_ip_visits
        #"model_accuracy": f"{accuracy * 100:.2f}%"
    }

    return jsonify(response_data)


@visit_blueprint.route("/test_results", methods=['GET', 'PATCH'])
def fetch_lab_results():
    """
    Endpoint to return lab results for a given MRN or all MRNs.
    Expects 'mrn' as an optional query parameter.
    """
    #   Extract MRN from query parameters (optional)
    mrn = request.args.get("mrn")

    # Get lab results from the model
    results, error = get_lab_results(mrn)

    if error:
        status_code = 404 if "No data found" in error else 500
        return jsonify({"error": error}), status_code

    return jsonify(results), 200