"""
API handler for phone call input.
Can be integrated with phone systems or used as a REST API endpoint.
"""
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from main_app import MentalHealthScreeningApp
from config import DATA_DIR, ARTIFACTS_DIR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')

# Initialize application
screening_app = None


def init_app():
    """Initialize the screening application."""
    global screening_app
    if screening_app is None:
        screening_app = MentalHealthScreeningApp()
    return screening_app


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Mental Health Screening API"}), 200


@app.route('/process_call', methods=['POST'])
def process_call():
    """
    Process a phone call recording.
    
    Expected form data:
    - file: Audio/video file (required)
    - soldier_id: Soldier identifier (required)
    - call_timestamp: Optional timestamp (ISO format)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400
        
        # Get soldier ID
        soldier_id = request.form.get('soldier_id')
        if not soldier_id:
            return jsonify({"error": "soldier_id is required"}), 400
        
        # Get optional timestamp
        call_timestamp_str = request.form.get('call_timestamp')
        call_timestamp = None
        if call_timestamp_str:
            try:
                call_timestamp = datetime.fromisoformat(call_timestamp_str)
            except ValueError:
                return jsonify({"error": "Invalid timestamp format. Use ISO format."}), 400
        
        # Save uploaded file
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize app if needed
        app_instance = init_app()
        
        # Process call
        result = app_instance.process_call(
            call_path=filepath,
            soldier_id=soldier_id,
            call_timestamp=call_timestamp
        )
        
        # Return results
        return jsonify({
            "success": True,
            "call_id": result["call_id"],
            "soldier_id": result["soldier_id"],
            "scores": result["scores"],
            "risk_assessment": {
                "severity": result["risk_assessment"]["severity"],
                "risk_factors": result["risk_assessment"]["risk_factors"],
                "requires_immediate_attention": result["risk_assessment"]["requires_immediate_attention"]
            },
            "alerts_count": len(result["alerts"]),
            "report_path": result["report_path"]
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/soldier_history/<soldier_id>', methods=['GET'])
def get_soldier_history(soldier_id):
    """Get screening history for a soldier."""
    try:
        app_instance = init_app()
        limit = request.args.get('limit', 10, type=int)
        history = app_instance.get_soldier_history(soldier_id, limit)
        return jsonify({"soldier_id": soldier_id, "history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get pending alerts."""
    try:
        app_instance = init_app()
        alerts = app_instance.get_pending_alerts()
        return jsonify({"alerts": alerts, "count": len(alerts)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert (mark as reviewed)."""
    # TODO: Implement alert acknowledgment in database
    return jsonify({"message": "Alert acknowledgment not yet implemented"}), 501


if __name__ == '__main__':
    # Initialize app on startup
    init_app()
    
    # Run Flask app
    # In production, use a proper WSGI server like gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)
