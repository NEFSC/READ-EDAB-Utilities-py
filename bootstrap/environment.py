# bootstrap/environment.py
import socket
import sys
from pathlib import Path
from datetime import datetime
import os

def get_path(dirs):
    for label, path in dirs.items():
        if os.path.exists(path):
            return Path(path)
    return None

def derive_dataset_path(resources_path: Path) -> Path:
    parts = resources_path.parts
    new_parts = []

    for part in parts:
        if part.lower() == "resources":
            # Preserve original casing
            replacement = "DATASETS" if part.isupper() else "Datasets" if part[0].isupper() else "datasets"
            new_parts.append(replacement)
        else:
            new_parts.append(part)

    dataset_path = Path(*new_parts)
    if not dataset_path.exists():
        raise FileNotFoundError(f"[BOOTSTRAP] Derived DATASETS path does not exist: {dataset_path}")
    return dataset_path

def bootstrap_environment(preferred=None, verbose=False):
    hostname = socket.gethostname()

    # 1. Explicitly map RESOURCES directoriess
    resources_root = {
        "NECMAC04363461.local": "/Users/kimberly.hyde/Documents/nadata/RESOURCES",
        "nefscsatdata": "/mnt/EDAB_Resources",
        "guihyde": "/mnt/EDAB_Resources",
        "Mac.localdomain": "/Users/kimberly.hyde/Documents/nadata/RESOURCES",
        "gdavis": "C:/Users/grace.davis/Documents/Hollings_2026/RESOURCES",
        "egable": "C:/Users/edmund.gable/Documents/GitHub/RESOURCES"
    }

    # 2. Explicitly map DATASETS directories
    datasets_root = {
        "NECMAC04363461.local": "/Users/kimberly.hyde/Documents/nadata/DATASETS",
        "nefscsatdata": "/mnt/EDAB_Datasets",
        "guihyde": "/mnt/EDAB_Datasets",
        "Mac.localdomain": "/Users/kimberly.hyde/Documents/nadata/DATASETS",
        "gdavis": "C:/Users/grace.davis/Documents/Hollings_2026/DATASETS",
        "egable": "C:/Users/edmund.gable/Documents/GitHub/DATASETS"
    }

    active_key = None
    root_path = None

    # Resolve RESOURCES root
    if preferred:
        if preferred in resources_root:
            candidate = Path(resources_root[preferred])
            if candidate.exists():
                root_path = candidate
                if verbose:
                    print(f"✓ Using specified RESOURCES directory: [{preferred}] → {candidate}")
            else:
                print(f"✗ Preferred RESOURCES path not found — falling back to defaults.")
        else:
            print(f"⚠ Unrecognized preferred label '{preferred}'. Valid options: {list(resources_root.keys())}")

    # If no preferred key (or it failed), try the hostname
    if root_path is None and hostname in resources_root:
        candidate = Path(resources_root[hostname])
        if candidate.exists():
            active_key = hostname
            root_path = candidate
            if verbose:
                print(f"✓ Using hostname RESOURCES directory: [{hostname}] → {candidate}")

    # Fallback: Check all remaining keys to see if any valid path exists on this machine
    if root_path is None:
        for key, path_str in resources_root.items():
            candidate = Path(path_str)
            if candidate.exists():
                active_key = key
                root_path = candidate
                if verbose:
                    print(f"✓ Using fallback RESOURCES directory: [{key}] → {candidate}")
                break

    if root_path is None:
        raise FileNotFoundError("[BOOTSTRAP] No valid RESOURCES directory found on this machine.")

    # --- Resolve DATASETS Root ---
    if active_key in datasets_root:
        dataset_path = Path(datasets_root[active_key])
        if not dataset_path.exists():
            raise FileNotFoundError(f"[BOOTSTRAP] Mapped DATASETS path does not exist: {dataset_path}")
    else:
        raise KeyError(f"[BOOTSTRAP] No dataset mapping found for key: '{active_key}'")

    # --- Resolve subdirectories ---
    python_path = root_path / "python"
    workflow_path = root_path / "workflow_resources"
    metadata_path = workflow_path / "metadata"
    lookup_path = workflow_path / "lookuptables"
    satlog_path = root_path / "logs/satprocessing"
    utilities_path = python_path / "utilities" / "src" / "utilities"

    # --- Validate existence of subdirectories ---
    for p in [python_path, workflow_path, metadata_path, lookup_path,satlog_path,utilities_path]:
        if not p.is_dir():
            raise FileNotFoundError(f"[BOOTSTRAP] Missing expected directory: {p}")

    # Add python path to sys.path
    if str(python_path) not in sys.path:
        sys.path.insert(0, str(python_path))

    if verbose:
        print(f"[BOOTSTRAP] Hostname: {hostname}")
        print(f"[BOOTSTRAP] Python path: {python_path}")
        print(f"[BOOTSTRAP] Utilities path: {utilities_path}")
        print(f"[BOOTSTRAP] Workflow resources: {workflow_path}")
        print(f"[BOOTSTRAP] Metadata path: {metadata_path}")
        print(f"[BOOTSTRAP] Look-up table path: {lookup_path}")
        print(f"[BOOTSTRAP] Dataset path: {dataset_path}")
        print(f"[BOOTSTRAP] Satprocessing logs path: {satlog_path}")
        print(f"[BOOTSTRAP] Timestamp: {datetime.now().isoformat()}")

    return {
        "hostname": hostname,
        "project_root": python_path,
        "python_path": python_path,
        "utilities_path": utilities_path,
        "workflow_resources": workflow_path,
        "metadata_path": metadata_path,
        "lookuptable_path": lookup_path,
        "dataset_path": dataset_path,
        "satlogs_path": satlog_path,
        "timestamp": datetime.now()
    }
