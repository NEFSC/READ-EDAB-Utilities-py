# utilities/notebook_utilities.py
from pathlib import Path
from .environment import bootstrap_environment, find_env_directory

def init_notebook_environment(verbose=False):
    """
    Dynamically resolves RESOURCES path based on the working directory
    and returns the bootstrapped environment.
    """
    # 1. Strictly use the directory where the notebook is running
    anchor_path = Path.cwd().resolve()

    # 2. Use your existing robust search tool
    resources_path = find_env_directory(anchor_path, ["resources", "edab_resources"])
    
    if not resources_path:
        raise OSError(f"[INIT] 'Resources' or 'EDAB_Resources' not found starting from: {anchor_path}")
        
    if verbose:
        print(f"[INIT] Dynamically resolved RESOURCES base path: {resources_path}")

    # 3. Call the main bootstrap script using the notebook's anchor
    return bootstrap_environment(start_path=anchor_path, verbose=verbose)