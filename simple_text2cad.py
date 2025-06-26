#!/usr/bin/env python3
"""
Simple Text2CAD Server - Generate STEP files from text prompts via API
"""

import os
import sys
import torch
import yaml
import tempfile
import logging
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
    
    # Load checkpoint if available
    if config["test"]["checkpoint_path"] is not None:
        checkpoint_file = config["test"]["checkpoint_path"]
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
    config_path = os.path.join(current_dir, "Cad_VLM/config/inference_user_input.yaml")
    try:
        model = load_model(config_path, device)
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
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Generate STEP file
            success = generate_step_file(model, prompt, temp_path, device)
            
            if not success:
                return jsonify({"error": "Failed to generate CAD model"}), 500
            
            # Verify file exists and has content
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return jsonify({"error": "Generated STEP file is empty or doesn't exist"}), 500
            
            # Create a safe filename for download
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            if len(safe_prompt) > 50:
                safe_prompt = safe_prompt[:50]
            
            download_filename = f"{safe_prompt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.step"
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=download_filename,
                mimetype='application/octet-stream'
            )
            
        finally:
            # Clean up temporary file after sending (Flask handles this automatically)
            pass
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    # Initialize model on startup
    logger.info("Initializing Text2CAD model...")
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting.")
        sys.exit(1)
    
    # Start the Flask app
    logger.info("Starting Simple Text2CAD API server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False) 