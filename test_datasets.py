import sys
import os
import json
from pathlib import Path

# Add project root to sys.path so we can import utilities
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

from utilities.dataset_utilities import (
    get_datasets_source,
    get_dataset_dirs,
    get_dataset_products,
    parse_dataset_info
)

def run_diagnostics(dataset_name):
    print("\n" + "="*60)
    print(f"🚀 DIAGNOSTICS FOR DATASET: {dataset_name}")
    print("="*60)

    print("\n--- STEP 1: Check Environment vs. Fallback Path ---")
    print(f"bootstrap_environment path: {env.get('dataset_path')}")
    try:
        source_path = get_datasets_source(verbose=True)
        print(f"get_datasets_source path:   {source_path}")
    except Exception as e:
        print(f"❌ Error in get_datasets_source: {e}")

    print(f"\n--- STEP 2: Check get_dataset_dirs('{dataset_name}') ---")
    try:
        dirs = get_dataset_dirs(dataset=dataset_name, verbose=True)
        print(json.dumps(dirs, indent=2))
    except Exception as e:
        print(f"❌ Error in get_dataset_dirs: {e}")

    print(f"\n--- STEP 3: Check get_dataset_products('{dataset_name}') ---")
    try:
        products = get_dataset_products(dataset=dataset_name, verbose=True)
        print(json.dumps(products, indent=2))
    except Exception as e:
        print(f"❌ Error in get_dataset_products: {e}")

    print("\n--- STEP 4: Linux Permission Check ---")
    # Let's manually check the server path just to be absolutely sure
    expected_path = f"/mnt/EDAB_Datasets/{dataset_name}"
    print(f"Checking path: {expected_path}")
    if os.path.exists(expected_path):
        print("✅ Path exists!")
        print(f"✅ Readable (r):  {os.access(expected_path, os.R_OK)}")
        print(f"✅ Executable (x): {os.access(expected_path, os.X_OK)} (Required for os.walk)")
        try:
            contents = os.listdir(expected_path)
            print(f"📂 Contents: {contents}")
        except Exception as e:
            print(f"❌ Could not list contents: {e}")
    else:
        print(f"❌ Path DOES NOT EXIST to the Python executor.")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # Grab the dataset name from the command line, default to OCCCI
    dataset_to_test = sys.argv[1].upper() if len(sys.argv) > 1 else 'OCCCI'
    run_diagnostics(dataset_to_test)
