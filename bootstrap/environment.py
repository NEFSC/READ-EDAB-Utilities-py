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

    # Map hostnames to RESOURCES root directories
    resources_root = {
        "khyde_laptop": "/Users/kimberly.hyde/Documents/nadata/RESOURCES/",
        "network": "/Volumes/EDAB_Resources/",
        "satdata": "/mnt/EDAB_Resources/",
        "container": "/mnt2/"
    }

    # Resolve RESOURCES root
    root_path = None
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

    if root_path is None:
        root_path = get_path(resources_root)
        if root_path and verbose:
            print(f"✓ Using default RESOURCES directory: {root_path}")

    if root_path is None:
        raise FileNotFoundError("No valid RESOURCES directory found.")

    # Derive DATASETS path by replacing 'RESOURCES' with 'DATASETS'
    dataset_path = derive_dataset_path(root_path)

    # Resolve subdirectories
    python_path = root_path / "python"
    workflow_path = root_path / "workflow_resources"
    metadata_path = workflow_path / "metadata"
    lookup_path = workflow_path / "lookuptables"
    satlog_path = root_path / "logs/satprocessing"

    # Validate existence
    for p in [python_path, workflow_path, metadata_path]:
        if not p.is_dir():
            raise FileNotFoundError(f"[BOOTSTRAP] Missing expected directory: {p}")

    # Add python path to sys.path
    if str(python_path) not in sys.path:
        sys.path.insert(0, str(python_path))

    if verbose:
        print(f"[BOOTSTRAP] Hostname: {hostname}")
        print(f"[BOOTSTRAP] Python path: {python_path}")
        print(f"[BOOTSTRAP] Workflow resources: {workflow_path}")
        print(f"[BOOTSTRAP] Metadata path: {metadata_path}")
        print(f"[BOOTSTRAP] Look-up table path: {lookup_path}")
        print(f"[BOOTSTRAP] Dataset path: {dataset_path}")
        print(f"[BOOTSTRAP] Satprocessing logs path: {satlog_path}")
        print(f"[BOOTSTRAP] Timestamp: {datetime.now().isoformat()}")

    return {
        "hostname": hostname,
        "project_root": python_path,
        "workflow_resources": workflow_path,
        "metadata_path": metadata_path,
        "lookuptable_path": lookup_path,
        "dataset_path": dataset_path,
        "satlogs_path": satlog_path,
        "timestamp": datetime.now()
    }