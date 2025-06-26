#!/usr/bin/env python3
"""
Simple Text2CAD Script - Generate STEP files from text prompts
No web interface, no complex setup - just text to STEP files.
"""

import os
import sys
import argparse
import torch
import yaml

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "Cad_VLM"))
sys.path.append(os.path.join(current_dir, "CadSeqProc"))

from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence


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
        print(f"Loading checkpoint: {checkpoint_file}")
        
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
    print(f"Generating CAD model for: '{text_prompt}'")
    
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cad_sequence.save_stp("generated_model", os.path.dirname(output_path))
        
        # Rename to desired output path
        generated_file = os.path.join(os.path.dirname(output_path), "generated_model.step")
        if os.path.exists(generated_file):
            os.rename(generated_file, output_path)
            print(f"✅ STEP file saved to: {output_path}")
            return True
        else:
            print("❌ Failed to generate STEP file")
            return False
            
    except Exception as e:
        print(f"❌ Error generating CAD model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate STEP files from text prompts")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="Text prompt describing the CAD model")
    parser.add_argument("--output", type=str, default="output.step", 
                       help="Output STEP file path (default: output.step)")
    parser.add_argument("--config", type=str, 
                       default="Cad_VLM/config/inference_user_input.yaml",
                       help="Config file path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.config, device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Generate STEP file
    success = generate_step_file(model, args.prompt, args.output, device)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 