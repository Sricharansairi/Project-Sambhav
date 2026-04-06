import subprocess
import sys
import os

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup():
    packages = [
        "torch",
        "sentence-transformers",
        "xgboost",
        "lightgbm",
        "scikit-learn",
        "joblib",
        "numpy",
        "pandas",
        "isotonic-calibration"
    ]
    
    print("--- Project Sambhav: Mastery Prediction Model Setup ---")
    
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"✅ {pkg} is already installed.")
        except ImportError:
            install_package(pkg)
            
    print("\n--- Downloading/Caching Local Embedding Model ---")
    try:
        from sentence_transformers import SentenceTransformer
        # This downloads the ~80MB model and caches it in ~/.cache/huggingface/hub
        model_name = "all-MiniLM-L6-v2"
        print(f"Loading {model_name}...")
        SentenceTransformer(model_name)
        print(f"✅ {model_name} is cached and ready.")
    except Exception as e:
        print(f"❌ Failed to cache embedding model: {e}")
        
    print("\n--- Setup Complete ---")

if __name__ == "__main__":
    setup()
