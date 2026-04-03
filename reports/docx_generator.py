import json
import os

def generate_docx(data, output_path):
    """Basic DOCX generator fallback."""
    with open(output_path, "w") as f:
        f.write("PROJECT SAMBHAV - DOCX REPORT\n")
        f.write("="*40 + "\n")
        f.write(json.dumps(data, indent=2))
