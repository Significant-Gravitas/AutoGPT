# organize_filesystem.py
# Temporary, may come in handly later

import json

from autogpt.llm_utils import call_ai_function

def reorganize():
    # Step 1: Evaluate the current file system
    file_system = evaluate_file_system()

    # Step 2: Generate a new organization structure
    new_structure = generate_new_structure(file_system)

    # Step 3: Reorganize the files according to the new structure
    reorganize_files(file_system, new_structure)

def evaluate_file_system():
    function = "evaluate_file_system"
    description = "Function to evaluate the current file system and return a list of files and folders (return type: List[str])."
    response = call_ai_function(function, [], description)
    file_system = parse_response(response)
    return file_system

def generate_new_structure(file_system):
    function = "generate_new_structure"
    description = "Function to generate a new organization structure based on the current file system (return type: Dict[str, List[str]])."
    response = call_ai_function(function, [file_system], description)
    new_structure = parse_response(response)
    return new_structure

def reorganize_files(file_system, new_structure):
    function = "reorganize_files"
    description = "Function to reorganize the files according to the new organization structure (return type: Dict[str, List[str]])."
    response = call_ai_function(function, [file_system, new_structure], description)
    reorganized_files = parse_response(response)
    return reorganized_files

def parse_response(response):
    try:
        parsed_data = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid response format: {response}")

    return parsed_data
