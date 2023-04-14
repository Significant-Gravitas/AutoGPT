# Test Runner Script

The test runner script is a Python script that executes tests for a command line application. Each test is defined in a separate directory in the `tests/prompt_tests` directory, and is defined by a `test.json` configuration file.

## Usage

The script accepts the following arguments:

- `--test`: Specifies which test or tests to run. Required. Valid values are `all` (to run all tests) or the name of a specific test subdirectory in `tests/prompt_tests`.
- `--list`: Lists the available tests and their names in the format `subdirectory_name: test_name`. Optional.

To run the script, use the following command in the root directory:

```
python test_runner.py --test [test_name|all] [--list]
```

## Creating Tests

To create your own tests, follow these steps:

1. Create a new directory in `tests/prompt_tests` with a descriptive name for your test.
2. Inside the test directory, create a `test.json` configuration file with the following fields:

   - `name`: The name of the test.
   - `description`: A description of what the test does.
   - `yaml_prompt`: The filename of the YAML prompt file to use for the test.
   - `output_files`: A list of filenames of the expected output files for the test.
   - `exec`: An object with the following fields:
   
     - `command`: The command to execute for the test. Should be the path to the executable file.
     - `arguments`: A list of command line arguments to pass to the executable.
     - `env` (optional): A dictionary of environment variables to set for the test.

     Example full `test.json` file:
     ```
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
     ```

3. Create the YAML prompt file for the test and save it in the test directory.
4. Create the expected output files for the test and save them in the test directory.
5. Run the test using the test runner script to ensure it is working as expected.