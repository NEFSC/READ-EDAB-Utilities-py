# src/utilities/environment.py
import os
import platform
import socket
import sys
from pathlib import Path
from datetime import datetime


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


# Assume find_env_directory is imported from your internal utilities
# from .helpers import find_env_directory

# Module-level dictionary to hold user-defined mount points
_MOUNT_POINTS = {
    "resources": "X://",
    "datasets": "V://",
    "projects": "Y://"
}

def configure_mount_points(resources=None, datasets=None, projects=None):
    """
    Externally configure base mount points before bootstrapping.
    Overrides dynamic searching. Useful for notebook use or custom pipeline wrappers.
    """
    if resources:
        _MOUNT_POINTS["resources"] = Path(resources).resolve()
    if datasets:
        _MOUNT_POINTS["datasets"] = Path(datasets).resolve()
    if projects:
        _MOUNT_POINTS["projects"] = Path(projects).resolve()

def _get_os_default_projects():
    """Returns a sensible default path for projects based on the operating system."""
    os_name = platform.system()
    if os_name == "Windows":
        return Path("C:/EDAB_Indicators/Projects")
    elif os_name == "Darwin": # macOS
        return Path("/Volumes/EDAB_Indicators/Projects")
    else: # Linux / Unix
        return Path("/mnt/EDAB_Indicators/khyde/Projects")

def bootstrap_environment(start_path=None, verbose=False):
    hostname = socket.gethostname()

    # 1. Determine anchor path
    if start_path:
        anchor_path = Path(start_path).resolve()
    else:
        anchor_path = Path.cwd().resolve()

    print(f"[BOOTSTRAP] Starting environment bootstrap from: {anchor_path}")

    # 2. Resolve RESOURCES
    # Check explicitly defined mount -> Env Var -> Dynamic Search
    if _MOUNT_POINTS["resources"]:
        resources_path = _MOUNT_POINTS["resources"]
    elif os.environ.get("EDAB_RESOURCES"):
        resources_path = Path(os.environ.get("EDAB_RESOURCES")).resolve()
    else:
        resources_path = find_env_directory(anchor_path, ["resources", "edab_resources"])
    
    if not resources_path or not resources_path.exists():
        raise FileNotFoundError(f"[BOOTSTRAP] Resources directory not found. Anchor: {anchor_path}")

    # 3. Resolve DATASETS
    if _MOUNT_POINTS["datasets"]:
        datasets_path = _MOUNT_POINTS["datasets"]
    elif os.environ.get("EDAB_DATASETS"):
        datasets_path = Path(os.environ.get("EDAB_DATASETS")).resolve()
    else:
        datasets_path = find_env_directory(anchor_path, ["datasets", "edab_dataset", "edab_datasets"])
        
    if not datasets_path:
        datasets_path = resources_path.parent / "DATASETS"
        datasets_path.mkdir(parents=True, exist_ok=True)
        if verbose: 
            print(f"⚠ DATASETS auto-created: {datasets_path}")

    # 4. Resolve PROJECTS
    if _MOUNT_POINTS["projects"]:
        projects_path = _MOUNT_POINTS["projects"]
    elif os.environ.get("EDAB_PROJECTS"):
        projects_path = Path(os.environ.get("EDAB_PROJECTS")).resolve()
    else:
        projects_path = find_env_directory(anchor_path, ["projects"])

    # Fallback for projects if not found
    if not projects_path:
        server_projects = _get_os_default_projects()
        if server_projects.exists():
            projects_path = server_projects
        else:
            fallback_path = resources_path.parent / "PROJECTS"
            try:
                fallback_path.mkdir(parents=True, exist_ok=True)
                projects_path = fallback_path
                if verbose: 
                    print(f"⚠ PROJECTS auto-created: {projects_path}")
            except PermissionError:
                projects_path = None
                if verbose: 
                    print("⚠ PROJECTS path not found and cannot be created (Permission Denied). Proceeding without it.")

    # 5. Resolve Operational Subdirectories inside Resources
    # Using relative pathlib joins works natively across all OS
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