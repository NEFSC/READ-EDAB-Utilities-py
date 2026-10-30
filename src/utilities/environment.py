# src/utilities/environment.py
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

def find_env_directory(anchor: Path, valid_names: list) -> Path:
    """Searches upwards and sideways for a directory matching valid_names."""
    targets = [name.lower() for name in valid_names]
    
    # Walk up the tree starting from the anchor
    for parent in [anchor] + list(anchor.parents):
        # 1. Is the current parent directory the target?
        if parent.name.lower() in targets:
            return parent
        
        # 2. Sibling search: Are any children of this parent the target?
        try:
            for child in parent.iterdir():
                if child.is_dir() and child.name.lower() in targets:
                    return child
        except PermissionError:
            # Safely ignore system directories we can't access during the search
            continue
            
    return None

def bootstrap_environment(start_path=None, verbose=False):
    hostname = socket.gethostname()

    # 1. Use the provided path, or default to where the pipeline/project script is being executed
    if start_path:
        anchor_path = Path(start_path).resolve()
    else:
        anchor_path = Path.cwd().resolve()

    # 2. Dynamically resolve RESOURCES root
    resources_path = find_env_directory(anchor_path, ["resources", "edab_resources"])
    if not resources_path:
        raise FileNotFoundError(f"[BOOTSTRAP] Resources directory not found relative to: {anchor_path}")

    datasets_path = find_env_directory(anchor_path, ["datasets", "edab_dataset", "edab_datasets"])
    if not datasets_path:
        datasets_path = resources_path.parent / "DATASETS"
        datasets_path.mkdir(parents=True, exist_ok=True)
        if verbose: print(f"⚠ DATASETS auto-created: {datasets_path}")

    projects_path = find_env_directory(anchor_path, ["projects"])
    if not projects_path:
        projects_path = resources_path.parent / "PROJECTS"
        projects_path.mkdir(parents=True, exist_ok=True)
        if verbose: print(f"⚠ PROJECTS auto-created: {projects_path}")

    # 2. Resolve Operational Subdirectories inside Resources
    operational_dirs = {
        "workflow_path": resources_path / "workflow_resources",
        "metadata_path": resources_path / "workflow_resources" / "metadata",
        "lookup_path": resources_path / "workflow_resources" / "lookuptables",
        "shapefile_path": resources_path / "workflow_resources" / "shapefiles",
        "satlog_path": resources_path / "logs" / "satprocessing"
    }

    for name, path in operational_dirs.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[BOOTSTRAP] Hostname: {hostname}")
        print(f"[BOOTSTRAP] Resources path: {resources_path}")
        print(f"[BOOTSTRAP] Dataset path: {datasets_path}")
        print(f"[BOOTSTRAP] Projects path: {projects_path}")
        print(f"[BOOTSTRAP] Timestamp: {datetime.now().isoformat()}")

    return {
        "hostname": hostname,
        "resources_path": resources_path,
        "dataset_path": datasets_path,
        "projects_path": projects_path,
        "workflow_resources": operational_dirs["workflow_path"],
        "metadata_path": operational_dirs["metadata_path"],
        "lookuptable_path": operational_dirs["lookup_path"],
        "shapefile_path": operational_dirs["shapefile_path"],
        "satlogs_path": operational_dirs["satlog_path"],
        "timestamp": datetime.now()
    }