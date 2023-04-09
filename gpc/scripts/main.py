import yaml
from src.refactoring_tool.code_analysis import *
from src.refactoring_tool.code_improvement import *
from src.refactoring_tool.test_generation import *
from src.refactoring_tool.code_refactoring import RefactoringOptions, refactor_code
from src.refactoring_tool.ai_functions import call_ai_function  # Import the function




# Read config file
with open("config/refactoring_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Define the input code using AI-generated code
function_name = "calculate"
args = ["a", "b"]
description = "Calculate the result of two numbers based on their relationship."

input_code = call_ai_function(function_name, args, description)

# Set refactoring options based on config file
options = RefactoringOptions(
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

# Refactor the code and generate tests
refactored_code, test_code = refactor_code(input_code, options)

# Print refactored code and generated tests
print("Refactored Code:")
print(refactored_code)

print("Generated Test Code:")
print(test_code)
