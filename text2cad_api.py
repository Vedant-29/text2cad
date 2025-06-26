from flask import Flask, request, jsonify, send_file
import subprocess
import os
import json
import tempfile
import shutil
from datetime import datetime
import glob
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_PATH = "/workspace/Text2CAD/config/inference_user_input.yaml"
SCRIPT_PATH = "/workspace/Text2CAD/Cad_VLM/test_user_input.py"
LOG_BASE_DIR = "/workspace/Text2CAD/logs"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Text2CAD API"})

@app.route('/generate-cad', methods=['POST'])
def generate_cad():
    """
    Generate CAD model from text prompt
    Expected JSON: {"prompt": "A simple cube"}
    Returns: STEP file as attachment
    """
    try:
        # Get prompt from request
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({"error": "Empty prompt provided"}), 400
        
        logger.info(f"Received request to generate CAD for prompt: {prompt}")
        
        # Change to the Cad_VLM directory
        os.chdir("/workspace/Text2CAD/Cad_VLM")
        
        # Run the Text2CAD model
        cmd = [
            "python3", "test_user_input.py",
            "--config_path", CONFIG_PATH,
            "--prompt", prompt
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return jsonify({
                "error": "Failed to generate CAD model",
                "details": result.stderr
            }), 500
        
        # Find the generated STEP file
        # The script outputs to logs with timestamp structure
        today = datetime.now().strftime("%Y-%m-%d")
        log_pattern = f"{LOG_BASE_DIR}/{today}/*/0/pred.step"
        
        step_files = glob.glob(log_pattern)
        if not step_files:
            logger.error(f"No STEP file found matching pattern: {log_pattern}")
            return jsonify({"error": "Generated STEP file not found"}), 500
        
        # Get the most recent file (in case multiple exist)
        step_file_path = max(step_files, key=os.path.getctime)
        logger.info(f"Found STEP file: {step_file_path}")
        
        # Verify file exists and has content
        if not os.path.exists(step_file_path) or os.path.getsize(step_file_path) == 0:
            return jsonify({"error": "Generated STEP file is empty or doesn't exist"}), 500
        
        # Create a safe filename for download
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        if len(safe_prompt) > 50:
            safe_prompt = safe_prompt[:50]
        
        download_filename = f"{safe_prompt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.step"
        
        return send_file(
            step_file_path,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/octet-stream'
        )
        
    except subprocess.TimeoutExpired:
        logger.error("Command timed out")
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/generate-cad-json', methods=['POST'])
def generate_cad_json():
    """
    Alternative endpoint that returns the STEP file content as base64 in JSON
    Useful for programmatic access
    """
    try:
        # Get prompt from request
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({"error": "Empty prompt provided"}), 400
        
        logger.info(f"Received JSON request to generate CAD for prompt: {prompt}")
        
        # Change to the Cad_VLM directory
        os.chdir("/workspace/Text2CAD/Cad_VLM")
        
        # Run the Text2CAD model
        cmd = [
            "python3", "test_user_input.py",
            "--config_path", CONFIG_PATH,
            "--prompt", prompt
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return jsonify({
                "error": "Failed to generate CAD model",
                "details": result.stderr
            }), 500
        
        # Find the generated STEP file
        today = datetime.now().strftime("%Y-%m-%d")
        log_pattern = f"{LOG_BASE_DIR}/{today}/*/0/pred.step"
        
        step_files = glob.glob(log_pattern)
        if not step_files:
            logger.error(f"No STEP file found matching pattern: {log_pattern}")
            return jsonify({"error": "Generated STEP file not found"}), 500
        
        # Get the most recent file
        step_file_path = max(step_files, key=os.path.getctime)
        logger.info(f"Found STEP file: {step_file_path}")
        
        # Read file content
        with open(step_file_path, 'rb') as f:
            file_content = f.read()
        
        # Convert to base64 for JSON response
        import base64
        file_base64 = base64.b64encode(file_content).decode('utf-8')
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "filename": os.path.basename(step_file_path),
            "file_size": len(file_content),
            "file_content_base64": file_base64,
            "generated_at": datetime.now().isoformat()
        })
        
    except subprocess.TimeoutExpired:
        logger.error("Command timed out")
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Make sure we're in the right conda environment
    import sys
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=8888, debug=False)