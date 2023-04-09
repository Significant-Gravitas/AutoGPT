import sys
import os
import openai
import yaml
import time
import argparse

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Import the analyze_code_performance function from your module
from src.refactoring_tool.code_analysis import CodeAnalysis

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

parser = argparse.ArgumentParser(description="Refactoring Tool")
parser.add_argument("-c", "--config", required=True, help="Path to the refactoring configuration file")
args = parser.parse_args()

def main():
    config_path = args.config

    # Load configuration
    config = load_config(config_path)
    openai.api_key = config["openai"]["api_key"]

    # Example Python code to analyze
    code_to_analyze = """
    import time

    def slow_function(n):
        result = 0
        for i in range(n):
            result += i
            time.sleep(0.1)
        return result

    print(slow_function(10))
    """

    # Call the analyze_code_performance function
    performance_suggestions = CodeAnalysis.analyze_code_performance(code_to_analyze)

    # Print the suggestions
    print("Performance suggestions:")
    for suggestion in performance_suggestions:
        print(f"- {suggestion}")

if __name__ == "__main__":
    main()
