import os
from pathlib import Path
from datetime import datetime
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)



def generate_logfile_path(pipeline_type, prods=None, dataset=None, env_dict=None):
    """
    Generates a standardized logfile path: satlogs_path/pipeline_type/YYYYMMDD_HHMMSS_pipeline_dataset_prods.log
    """
           
    if env_dict is None:
        from utilities.bootstrap.environment import bootstrap_environment
        env_dict = env
    
    base_log_dir = env_dict.get('satlogs_path')
    if not base_log_dir:
        # Fallback if satlogs_path is missing from the environment
        base_log_dir = Path.cwd() / "logs" / "satprocessing"
        
    # Create the pipeline-specific subdirectory (e.g., 'anomaly' or 'stats')
    log_dir = Path(base_log_dir) / pipeline_type.lower()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Format the filename components
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Handle optional products and dataset strings safely
    if prods:
        prod_str = f"_{'-'.join(prods) if isinstance(prods, list) else str(prods)}"
    else:
        prod_str = ""

    ds_str = f"-{dataset}" if dataset else ""
    
    filename = f"{timestamp}_{pipeline_type}{ds_str}_{prod_str}.log"
    return str(log_dir / filename)