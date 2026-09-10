import os

# Environment variables to eliminate CPU multi-thread contention and heating
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import random
import yaml
import numpy as np
import torch

# Limit PyTorch CPU threads to 1 so CPU stays cool while GPU handles 100% compute
torch.set_num_threads(1)

def load_config(config_path="config.yaml"):
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        # Fallback if executed from subfolder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_device():
    """Returns PyTorch device safely testing CUDA kernel execution."""
    if torch.cuda.is_available():
        try:
            # Test Conv2d kernel execution to verify sm_120 GPU architecture binary support
            conv = torch.nn.Conv2d(3, 16, 3).cuda()
            dummy = torch.randn(1, 3, 32, 32, device="cuda")
            _ = conv(dummy)
            print(f"[Device Info] Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"[Device Info] GPU capability warning ({e}). Running reliably on CPU.")
            return torch.device("cpu")
    print("[Device Info] Using CPU device.")
    return torch.device("cpu")

def ensure_dirs(dirs_list):
    """Ensures specified directories exist."""
    for d in dirs_list:
        os.makedirs(d, exist_ok=True)

def save_json(data, filepath):
    """Saves dictionary data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """Loads JSON file into dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Saves PyTorch model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics
    }, filepath)
