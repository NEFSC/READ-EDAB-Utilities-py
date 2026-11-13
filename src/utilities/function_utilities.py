import os
import ast
import re
import importlib.util
import importlib.metadata
import json
import sys
from pathlib import Path

""" 
Purpose:
    FUNCTION_UTILITIES is a collection of utility functions for working with the python functions in utilities.

Main Functions:
    - get_pyfile_functions: Gets a list of all functions in utilities (or a specified directory)
    - find_function: Searches through the python function names for a specific text string
    - find_text: Searches through the python functions for specific text strings
    - get_installed_packages: Returns a dictionary of installed packages and their installation source.

Helper Functions:
    - wildcard_to_regex: Converts wildcard text string to a regex text string
    - get_package_path: Dynamically locates the physical directory of an installed Python package.
    
Copywrite: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on September 18, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Sep 18, 2025 - KJWH: Moved get_pyfile_functions from file_utilities to function_utitlities
                         Create new find_text to search functions for a text string
    Nov 13, 2026 - KJWH: Updated functions to work with the new utilities and satdata_processing packages instead of specified directories. 
                         Added get_package_path to find the physical directory of an installed Python package
                         Added get_installed_packages to find all installed packages
                         Now using get_installed_packages to find the local editable packages instead of hardcoding the package names in get_pyfile_functions, find_function, and find_text
"""

def wildcard_to_regex(pattern, case_insensitive=True):
    escaped = re.escape(pattern).replace(r'\*', '.*')
    flags = re.IGNORECASE if case_insensitive else 0
    return re.compile(escaped, flags=flags)

def get_package_path(package_name: str) -> Path:
    """Dynamically locates the physical directory of an installed Python package."""
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError(f"Cannot find installed package: {package_name}. Is it installed in this environment?")
    return Path(spec.submodule_search_locations[0])

def get_installed_packages(search_string=None, custom_only=False):
    """
    Returns a dictionary of installed packages and their installation source.
    If custom_only=True, it only returns GitHub or Local Editable packages.
    """
    packages = {}
    
    # Get the absolute root path of the active Python/Conda environment
    env_prefix = Path(sys.prefix).resolve()
    
    for dist in importlib.metadata.distributions():
        if "Name" not in dist.metadata:
            continue
            
        name = dist.metadata["Name"]
        install_source = "Standard"
        
        # 1. Primary Check: pip's strict metadata
        direct_url_content = dist.read_text("direct_url.json")
        if direct_url_content:
            try:
                url_data = json.loads(direct_url_content)
                if "vcs_info" in url_data and url_data["vcs_info"].get("vcs") == "git":
                    install_source = "GitHub"
                elif "dir_info" in url_data and url_data["dir_info"].get("editable") is True:
                    install_source = "Local Editable"
            except json.JSONDecodeError:
                pass

        # 2. Foolproof Fallback Check: Physical path boundary
        if install_source == "Standard":
            try:
                # Get the absolute path where the package code lives
                pkg_path = Path(dist.locate_file("")).resolve()
                
                # If the package lives completely outside the Conda environment folder, it's custom.
                if not str(pkg_path).startswith(str(env_prefix)):
                    install_source = "Local Editable"
            except Exception:
                pass

        if custom_only and install_source == "Standard":
            continue
            
        packages[name] = install_source

    # Apply optional wildcard search
    if search_string:
        regex = wildcard_to_regex(search_string, case_insensitive=True)
        packages = {k: v for k, v in packages.items() if regex.search(k)}
        
    return dict(sorted(packages.items()))

def get_pyfile_functions(package_names=None, search_string=None):
    """
    Retrieves all functions from the specified packages. 
    If package_names is None, it defaults to all custom installed packages.
    """
    function_map = {}
    
    # Default to all custom packages if none are specified
    if package_names is None:
        package_names = list(get_installed_packages(custom_only=True).keys())
    # Allow a single string to be passed instead of a list
    elif isinstance(package_names, str):
        package_names = [package_names]

    for pkg in package_names:
        try:
            directory = get_package_path(pkg)
            print(f"Searching package: {pkg} at {directory}")
        except ImportError as e:
            print(f"Skipping {pkg}: {e}")
            continue

        # Recursively find all .py files
        for file_path in directory.rglob("*.py"):
            filename = file_path.name
            
            if filename == "__init__.py":
                continue
            if search_string and search_string not in filename:
                continue

            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    source = file.read()
                    node = ast.parse(source, filename=filename)
                    
                    functions = [
                        n.name for n in node.body
                        if isinstance(n, ast.FunctionDef)
                    ]
                    
                    if functions:
                        module_name = file_path.relative_to(directory).with_suffix("").as_posix()
                        function_map[file_path] = {
                            "package": pkg, 
                            "module_name": module_name, 
                            "functions": functions
                        }
                except SyntaxError:
                    continue

    return function_map

def find_function(search_string, package_names=None):
    """Searches for a specific function name across specified or custom packages."""
    function_map = get_pyfile_functions(package_names=package_names)
    regex = wildcard_to_regex(search_string, case_insensitive=True)

    results = []
    for file_path, data in function_map.items():
        pkg = data["package"]
        module_name = data["module_name"]
        
        for func_name in data["functions"]:
            if regex.search(func_name):
                results.append({
                    "package": pkg,
                    "module": module_name,
                    "function": func_name
                })

    return results

def find_text(search_string, package_names=None):
    """Searches for a text string inside the functions of specified or custom packages."""
    function_map = get_pyfile_functions(package_names=package_names)
    regex = wildcard_to_regex(search_string, case_insensitive=True)
    print(f"Regex for string '{search_string}' is: {regex.pattern}")

    results = []
    for file_path, data in function_map.items():
        pkg = data["package"]
        module_name = data["module_name"]
        functions = data["functions"]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
                tree = ast.parse(source, filename=str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name in functions:
                        start_line = node.lineno
                        end_line = getattr(node, 'end_lineno', None)

                        if end_line is None:
                            end_line = max(
                                [n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')],
                                default=start_line
                            )

                        func_lines = source.splitlines()[start_line - 1:end_line]

                        for i, line in enumerate(func_lines, start=start_line):
                            if regex.search(line.strip()):
                                results.append({
                                    "package": pkg,
                                    "module": module_name,
                                    "function": node.name,
                                    "line_number": i,
                                    "line": line.strip()
                                })

        except Exception as e:
            results.append({
                "package": pkg,
                "module": module_name,
                "error": str(e)
            })

    return results