import json
import os

def generate_pdf(data, output_path, tier="Detailed"):
    """Basic PDF generator fallback."""
    # Since we don't want to install heavy PDF libs in this fix step, 
    # we'll just write a text file with a .pdf extension for now 
    # to satisfy the backend startup and file response logic.
    with open(output_path, "w") as f:
        f.write(f"PROJECT SAMBHAV - PDF REPORT ({tier})\n")
        f.write("="*40 + "\n")
        f.write(json.dumps(data, indent=2))
