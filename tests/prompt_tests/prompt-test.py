"""
Write a python script that executes tests.
- Each test is in a subdirectory in tests/prompt_tests
- The script accepts the arguments: "--test", "--list"
- The argument "--test" is required and specifies either "all" for all tests or a specific subdirectory name for a single test
- The argument "--list" is optional and asks the script to list the tests that are available using the format "subdirectory_name: test_name"
- When "--test all" is specified, the script executes all tests, e.g. loading the test from each subdirectory.
- Each test has a config file in the subdirectory called test.json
- A function called run_test(test_subdirectory) is called for each test
- The run_test() function loads "{test_subdirectory}/test.json" and executes the test
- The "exec" field in the test.json file specifies the command to execute and the arguments to pass to the command. The command is called with the python3 interpreter.
Example test.json:
{
    "name": "Denver Weather Next Week",
    "description": "Get the weather over the next week in Denver, CO. The information returned is not likely to be accurate, however, the task is completed.",
    "yaml_prompt": "weather-denver-txt.yaml",
    "output_files": [
        "weather-denver.txt"
    ],
    "exec": {
        "command": "scripts/main.py",
        "arguments": [
            "--continuous",
            "--continuous-limit",
            "20",
            "--use-yaml-file"
        ],
        "env": {
            "TEMPERATURE": "0"
        }
    }
}
"""
import argparse
import os
import json
import subprocess

def run_test(test_subdirectory):
    test_path = os.path.join("tests", "prompt_tests", test_subdirectory)
    config_path = os.path.join(test_path, "test.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    command = ["python3", config["exec"]["command"]] + config["exec"]["arguments"]
    env = os.environ.copy()
    env.update(config["exec"].get("env", {}))
    subprocess.run(command, cwd=test_path, env=env)

def list_tests():
    for test_subdirectory in os.listdir(os.path.join("tests", "prompt_tests")):
        config_path = os.path.join("tests", "prompt_tests", test_subdirectory, "test.json")
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        print(f"{test_subdirectory}: {config['name']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, choices=["all"] + os.listdir(os.path.join("tests", "prompt_tests")))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_tests()
        return

    if args.test == "all":
        for test_subdirectory in os.listdir(os.path.join("tests", "prompt_tests")):
            run_test(test_subdirectory)
    else:
        run_test(args.test)

if __name__ == "__main__":
    main()
