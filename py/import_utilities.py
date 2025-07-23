import os
import sys
import socket
import ast
import importlib.util
import types

def get_python_dir():
    # 1. Identify the computer by hostname
    hostname = socket.gethostname()
    #print(f"Running on: {hostname}")

    # 2. Set default Python code location based on hostname
    code_locations = {                                              # 2. Set default Python code location based on hostname
        "NECMAC04363461.local": "/Users/kimberly.hyde/Documents/",  # Mac laptop
        "nefscsatdata": "/mnt/EDAB_Archive/",
        "guihyde": "/mnt/EDAB_Archive/"
    }

    # 3. Set default path to the python utilties functions
    default_code_path = os.path.join(code_locations.get(hostname),"nadata/python/utilities/py")    
    if not os.path.isdir(default_code_path):
        print(f"Directory not found: {default_code_path}")
        return {}
    return default_code_path

    
def get_pyfile_functions(directory=None,search_string=None):
    function_map = {}

    # Get the directory if not provided
    if not directory:
        directory = get_python_dir()
    if not directory:
        return {}

    # Find all files in the directory
    for filename in os.listdir(directory):
        # Look for .py files
        if filename.endswith(".py") and filename != "__init__.py":

            # Look for specific files if search_string provided
            if search_string and search_string not in filename:
                continue

            file_path = os.path.join(directory, filename)

            with open(file_path, "r") as file:
                node = ast.parse(file.read(), filename=filename)

                functions = [
                    n.name for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]

            module_name = filename[:-3]  # Strip .py
            function_map[module_name] = functions

    return function_map

def import_utility_functions(directory=None,function_map=None):

    # Get the default directory if not provided
    if not directory:
        directory = get_python_dir()
    if not directory:
        return {}

    # Get the default function_map if not provided
    if not function_map:
        function_map = get_pyfile_functions(directory)
    if not function_map:
        return {}

    imported_functions = {}
    for module_name, function_names in function_map.items():
        module_path = os.path.join(directory, f"{module_name}.py")
        if not os.path.isfile(module_path):
            continue

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for name in function_names:
                func = getattr(module, name, None)
                if isinstance(func, types.FunctionType):
                    imported_functions[name] = func
                    #print(f"Imported: {name} from {module_name}")
    
    return imported_functions

