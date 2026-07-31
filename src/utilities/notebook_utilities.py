# utilities/notebook_utilities.py

import socket
import sys
from pathlib import Path

def init_notebook_environment(verbose=False):
    """
    Dynamically resolves RESOURCES path based on file or working directory,
    adds python path to sys.path, and returns the bootstrapped environment.
    """

    # 1. Determine script/notebook path
    try:
        current_file = Path(__file__).resolve()
    except NameError:
        # Crucial fallback for interactive environments like Jupyter Notebooks
        current_file = Path.cwd().resolve()

    # 2. Dynamically resolve RESOURCES root
    resource_candidates = [
        p for p in [current_file] + list(current_file.parents)
        if p.name.lower() in ["resources", "edab_resources"]
    ]

    if not resource_candidates:
        raise EnvironmentError(f"[INIT] 'Resources' or 'EDAB_Resources' not found in path: {current_file}")
    elif len(resource_candidates) > 1:
        raise ValueError(f"[INIT] Multiple Resources directories found in path, which is ambiguous: {resource_candidates}")

    base_path = resource_candidates[0]
    
    if verbose:
        print(f"[INIT] Dynamically resolved RESOURCES base path: {base_path}")
    
    # 3. Resolve python root and add to sys.path
    project_root = base_path / "python"
    if not project_root.is_dir():
        raise FileNotFoundError(f"[INIT] Project root not found: {project_root}")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 4. Import and run the main bootstrap script
    try:
        from utilities.bootstrap.environment import bootstrap_environment
    except ModuleNotFoundError as e:
        raise ImportError(f"[INIT] Failed to import bootstrap_environment: {e}")

    # Pass the preferred argument down to maintain consistency
    return bootstrap_environment(verbose=verbose)

