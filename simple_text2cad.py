#!/usr/bin/env python3
"""
Simple Text2CAD Server - Generate STEP files from text prompts via API

Environment variables:
    TEXT2CAD_CHECKPOINT  Path to Text2CAD_1.0.pth. Falls back to
                         test.checkpoint_path in the inference config.
    PORT                 Port to listen on (default 5000).
"""

import os
import sys
import torch
import yaml
import io
import shutil
import tempfile
import logging
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_file

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "Cad_VLM"))
sys.path.append(os.path.join(current_dir, "CadSeqProc"))

from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
device = None

# One generation at a time: the model and GPU are shared across requests
generate_lock = threading.Lock()

CONFIG_PATH = os.path.join(current_dir, "Cad_VLM", "config", "inference_user_input.yaml")


def load_model(config_path, device):
    """Load the Text2CAD model"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Configure CAD decoder
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    
    # Initialize model
    text2cad = Text2CAD(
        text_config=config["text_encoder"], 
        cad_config=cad_config
    ).to(device)
    
    # Checkpoint: TEXT2CAD_CHECKPOINT env var, else the value in the YAML config
    checkpoint_file = os.environ.get("TEXT2CAD_CHECKPOINT") or config["test"]["checkpoint_path"]
    if checkpoint_file is not None:
        checkpoint_file = os.path.expanduser(checkpoint_file)
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_file}. "
                "Set TEXT2CAD_CHECKPOINT or test.checkpoint_path in the config."
            )
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        
        checkpoint = torch.load(checkpoint_file, map_location=device)
        pretrained_dict = {}
        
        # Handle module prefixes
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value
        
        text2cad.load_state_dict(pretrained_dict, strict=False)
    
    text2cad.eval()
    return text2cad


def generate_step_file(model, text_prompt, output_path, device):
    """Generate STEP file from text prompt"""
    logger.info(f"Generating CAD model for: '{text_prompt}'")
    
    # Generate CAD sequence
    with torch.no_grad():
        pred_cad_seq_dict = model.test_decode(
            texts=[text_prompt],
            maxlen=MAX_CAD_SEQUENCE_LENGTH,
            nucleus_prob=0,
            topk_index=1,
            device=device,
        )
    
    try:
        # Convert to CAD sequence and save as STEP
        cad_sequence = CADSequence.from_vec(
            pred_cad_seq_dict["cad_vec"][0].cpu().numpy(),
            bit=N_BIT,
            post_processing=True,
        )
        
        # Save as STEP file
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there's a directory component
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = "."  # Use current directory if no directory specified
        
        cad_sequence.save_stp("generated_model", output_dir)
        
        # Rename to desired output path
        generated_file = os.path.join(output_dir, "generated_model.step")
        if os.path.exists(generated_file):
            os.rename(generated_file, output_path)
            logger.info(f"✅ STEP file saved to: {output_path}")
            return True
        else:
            logger.error("❌ Failed to generate STEP file")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error generating CAD model: {e}")
        return False


def initialize_model():
    """Initialize the model on server startup"""
    global model, device
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(CONFIG_PATH, device)
        logger.info("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "Simple Text2CAD API",
        "model_loaded": model is not None
    })


@app.route('/generate-cad', methods=['POST'])
def generate_cad():
    """
    Generate CAD model from text prompt
    Expected JSON: {"prompt": "A simple cube"}
    Returns: STEP file as attachment
    """
    global model, device
    
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded")
        return jsonify({"error": "Model not loaded. Please check server startup logs."}), 500
    
    try:
        # Get prompt from request
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({"error": "Empty prompt provided"}), 400
        
        logger.info(f"Received request to generate CAD for prompt: {prompt}")
        
        # Each request gets its own temporary directory, removed after reading
        temp_dir = tempfile.mkdtemp(prefix="text2cad_")
        temp_path = os.path.join(temp_dir, "output.step")
        
        try:
            # Generate STEP file
            with generate_lock:
                success = generate_step_file(model, prompt, temp_path, device)
            
            if not success:
                return jsonify({"error": "Failed to generate CAD model"}), 500
            
            # Verify file exists and has content
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return jsonify({"error": "Generated STEP file is empty or doesn't exist"}), 500
            
            with open(temp_path, "rb") as f:
                step_bytes = f.read()
            
            # Create a safe filename for download
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            if len(safe_prompt) > 50:
                safe_prompt = safe_prompt[:50]
            
            download_filename = f"{safe_prompt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.step"
            
            return send_file(
                io.BytesIO(step_bytes),
                as_attachment=True,
                download_name=download_filename,
                mimetype='application/octet-stream'
            )
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    # Initialize model on startup
    logger.info("Initializing Text2CAD model...")
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting.")
        sys.exit(1)
    
    # Start the Flask app (never in debug mode: it allows remote code execution)
    port = int(os.environ.get("PORT", "5000"))
    logger.info(f"Starting Simple Text2CAD API server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
