import json
import os

def generate_xlsx(data, output_path):
    """Basic XLSX generator fallback."""
    with open(output_path, "w") as f:
        f.write("PROJECT SAMBHAV - XLSX REPORT\n")
        f.write("="*40 + "\n")
        f.write(json.dumps(data, indent=2))
