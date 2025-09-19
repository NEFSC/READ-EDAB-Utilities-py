import os
import ast
import re
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

"""
Purpose:
    FUNCTION_UTILITIES is a collection of utility functions for working with the python functions in utilities.

Main Functions:
    - get_pyfile_functions: Gets a list of all functions in utilities (or a specified directory)
    - find_function: Searches through the python function names for a specific text string
    - find_text: Searches through the python functions for specific text strings

Helper Functions:
    - wildcard_to_regex: Converts wildcard text string to a regex text string
    
Copywrite: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on September 18, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Sep 18, 2025 - KJWH: Moved get_pyfile_functions from file_utilities to function_utitlities
                         Create new find_function_text to search functions for a text string
"""

def wildcard_to_regex(pattern, case_insensitive=True):
    escaped = re.escape(pattern).replace(r'\*', '.*')
    flags = re.IGNORECASE if case_insensitive else 0
    return re.compile(escaped, flags=flags)

def get_pyfile_functions(directory=None,search_string=None):
    from utilities.bootstrap.environment import bootstrap_environment
    env = bootstrap_environment(verbose=False)

    function_map = {}

    # Get the directory if not provided
    if not directory:
        directory = env["utilities_path"]
        print(f"directory: {directory}")
    if not directory:
        raise ValueError("No valid directory provided and 'utilities_path' is undefined.")

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

def find_function(search_string, directory=None):
    function_map = get_pyfile_functions(directory=directory)
    regex = wildcard_to_regex(search_string, case_insensitive=True)

    results = []

    for module_name, functions in function_map.items():
        for func_name in functions:
            if regex.search(func_name):
                results.append({
                    "module": module_name,
                    "function": func_name
                })

    return results

def find_text(search_string,directory=None):
    
    if not directory:
        directory = env["utilities_path"]

    function_map = get_pyfile_functions(directory=directory)
    
    regex = wildcard_to_regex(search_string, case_insensitive=True)
    print(f"regex for string {search_string} is: {regex}")


    results = []
    for module_name, functions in function_map.items():
        file_path = os.path.join(directory, f"{module_name}.py")
        if not os.path.exists(file_path):
            results.append({
                "module": module_name,
                "error": f"File not found: {file_path}"
            })
            continue

        try:
            with open(file_path, "r") as f:
                source = f.read()
                tree = ast.parse(source, filename=file_path)

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
                            normalized_line = line.strip()
                            if regex.search(normalized_line):  # Single-line match only
                                results.append({
                                    "module": module_name,
                                    "function": node.name,
                                    "line_number": i,
                                    "line": line.strip()
                                })

        except Exception as e:
            results.append({
                "module": module_name,
                "error": str(e)
            })

    return results