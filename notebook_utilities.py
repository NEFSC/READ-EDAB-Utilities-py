# utilities/notebook_utilities.py

import socket
import sys
from pathlib import Path

def init_notebook_environment(verbose=False):
    """
    Dynamically resolves RESOURCES path based on hostname,
    adds python path to sys.path, and returns the bootstrapped environment.
    """
    hostname = socket.gethostname()

    resources_root = {
        "NECMAC04363461.local": "/Users/kimberly.hyde/Documents/nadata/RESOURCES",
        "nefscsatdata": "/mnt/EDAB_Resources",
        "guihyde": "/mnt/EDAB_Resources"
    }

    base_path = resources_root.get(hostname)
    if not base_path:
        raise EnvironmentError(f"[INIT] Unknown hostname: {hostname}")

    project_root = Path(base_path) / "python"
    if not project_root.is_dir():
        raise FileNotFoundError(f"[INIT] Project root not found: {project_root}")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from utilities.bootstrap.environment import bootstrap_environment
    except ModuleNotFoundError as e:
        raise ImportError(f"[INIT] Failed to import bootstrap_environment: {e}")

    return bootstrap_environment(verbose=verbose)