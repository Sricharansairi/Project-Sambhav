import json
import os

def generate_pptx(data, output_path):
    """Basic PPTX generator fallback."""
    with open(output_path, "w") as f:
        f.write("PROJECT SAMBHAV - PPTX REPORT\n")
        f.write("="*40 + "\n")
        f.write(json.dumps(data, indent=2))
