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

def bootstrap_environment(verbose=False):
    hostname = socket.gethostname()

    # 1. Determine script path
    # Use __file__ to get the absolute path of the current script.
    # There is a fallback to cwd() in case this is run in an interactive environment like Jupyter.
    try:
        current_file = Path(__file__).resolve()
    except NameError:
        current_file = Path.cwd().resolve()

    # 2. Dynamically resolve RESOURCES root
    # Search up the path tree for a directory named "Resources" or "EDAB_Resources" (case-insensitive)
    resource_candidates = [
        p for p in [current_file] + list(current_file.parents)
        if p.name.lower() in ["resources", "edab_resources"]
    ]

    if not resource_candidates:
        raise FileNotFoundError(f"[BOOTSTRAP] 'Resources' or 'EDAB_Resources' not found in path: {current_file}")
    elif len(resource_candidates) > 1:
        raise ValueError(f"[BOOTSTRAP] Multiple Resources directories found in path, which is ambiguous: {resource_candidates}")

    root_path = resource_candidates[0]

    if verbose:
        print(f"✓ Dynamically resolved RESOURCES directory: {root_path}")

    # 3. Dynamically resolve DATASETS root (and create if missing)
    # Look in the directory exactly one level above the RESOURCES root
    base_dir = root_path.parent
    dataset_candidates = [
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.lower() in ["datasets", "edab_datasets"]
    ]

    if not dataset_candidates:
        # If no datasets folder exists, automatically create a standard one
        dataset_path = base_dir / "DATASETS"
        dataset_path.mkdir(parents=True, exist_ok=True)
        print(f"⚠ DATASETS directory not found. Auto-created: {dataset_path}")
    elif len(dataset_candidates) > 1:
        raise ValueError(f"[BOOTSTRAP] Multiple DATASETS directories found in {base_dir}, which is ambiguous: {dataset_candidates}")
    else:
        dataset_path = dataset_candidates[0]
        if verbose:
            print(f"✓ Dynamically resolved DATASETS directory: {dataset_path}")


    # --- Resolve required source code subdirectories ---
    python_path = root_path / "python"
    utilities_path = python_path / "utilities" / "src" / "utilities"

    # --- Validate existence of source subdirectories ---
    for p in [python_path, utilities_path]:
        if not p.is_dir():
            raise FileNotFoundError(f"[BOOTSTRAP] Missing essential source directory (cannot auto-create): {p}")

    # --- Resolve operational subdirectories (to be auto-created if missing) ---
    operational_dirs = {
        "workflow_path": root_path / "workflow_resources",
        "metadata_path": root_path / "workflow_resources" / "metadata",
        "lookup_path": root_path / "workflow_resources" / "lookuptables",
        "satlog_path": root_path / "logs" / "satprocessing"
    }

    for name, path in operational_dirs.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"⚠ Missing operational directory auto-created: {path}")

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
        "workflow_resources": operational_dirs["workflow_path"],
        "metadata_path": operational_dirs["metadata_path"],
        "lookuptable_path": operational_dirs["lookup_path"],
        "dataset_path": dataset_path,
        "satlogs_path": operational_dirs["satlog_path"],
        "timestamp": datetime.now()
    }
