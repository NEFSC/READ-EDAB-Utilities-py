# utilities/notebook_utilities.py

import socket
import sys
from pathlib import Path

def init_notebook_environment(preferred=None, verbose=False):
    """
    Dynamically resolves RESOURCES path based on hostname,
    adds python path to sys.path, and returns the bootstrapped environment.
    """
    hostname = socket.gethostname()

    resources_root = {
        "khyde_laptop": "/Users/kimberly.hyde/Documents/nadata/RESOURCES/",
        "gdavis": r"C:\\Users\\grace.davis\\Documents\\Hollings_2026\\RESOURCES\\",
        "egable": r"C:\\Users\\edmund.gable\\Documents\\GitHub\\RESOURCES\\",
        "network": "/Volumes/EDAB_Resources/",
        "satdata": "/mnt/EDAB_Resources/",
        "hsynan":r"\\nefscdata\EDAB_Resources",
        "container": "/mnt2/"
    }

    base_path = None

    # 1. Try the preferred key first
    if preferred and preferred in resources_root:
        candidate = Path(resources_root[preferred])
        if candidate.exists():
            base_path = candidate

    # 2. Try the hostname
    if not base_path and hostname in resources_root:
        candidate = Path(resources_root[hostname])
        if candidate.exists():
            base_path = candidate

    # 3. Fallback: check all mapped paths to see if any exist locally
    if not base_path:
        for path_str in resources_root.values():
            candidate = Path(path_str)
            if candidate.exists():
                base_path = candidate
                break

    if not base_path:
        raise EnvironmentError(f"[INIT] Could not find a valid RESOURCES base path on this machine.")

    # Resolve python root and add to sys.path
    project_root = base_path / "python"
    if not project_root.is_dir():
        raise FileNotFoundError(f"[INIT] Project root not found: {project_root}")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import and run the main bootstrap script
    try:
        from utilities.bootstrap.environment import bootstrap_environment
    except ModuleNotFoundError as e:
        raise ImportError(f"[INIT] Failed to import bootstrap_environment: {e}")

    # Pass the preferred argument down to maintain consistency
    return bootstrap_environment(preferred=preferred, verbose=verbose)

