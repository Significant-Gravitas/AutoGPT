import sys
import os
import openai
import yaml
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.refactoring_tool.code_refactoring import RefactoringOptions, refactor_code



# from src.refactoring_tool.ai_functions import call_ai_function
# from src.refactoring_tool.code_refactoring import RefactoringOptions, refactor_code


def load_configuration(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def get_refactoring_options(config):
    return RefactoringOptions(
        analyze_readability=config["analyze_options"]["readability"],
        analyze_performance=config["analyze_options"]["performance"],
        analyze_security=config["analyze_options"]["security"],
        analyze_modularity=config["analyze_options"]["modularity"],
        refactor_variables=config["refactoring_options"]["refactor_variables"],
        optimize_loops=config["refactoring_options"]["optimize_loops"],
        simplify_conditionals=config["refactoring_options"]["simplify_conditionals"],
        refactor_functions=config["refactoring_options"]["refactor_functions"],
        generate_unit_tests=config["testing_options"]["unit_tests"],
        generate_integration_tests=config["testing_options"]["integration_tests"],
        generate_system_tests=config["testing_options"]["system_tests"],
    )


config_path = "config/refactoring_config.yaml"
config = load_configuration(config_path)
openai.api_key = config["openai"]["api_key"]

function_name = "calculate"
args = ["a", "b"]
description = "Calculate the result of two numbers based on their relationship."

input_code = call_ai_function(function_name, tuple(args), description)

options = get_refactoring_options(config)

refactored_code, test_code = refactor_code(input_code, options)

print("Refactored Code:")
print(refactored_code)

print("Generated Test Code:")
print(test_code)
