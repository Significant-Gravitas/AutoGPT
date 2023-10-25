# Main Script
The `main.py` script is the entry point of the application. It sets up the AI model and the databases, and then runs a series of steps based on the provided configuration. The script uses the Typer library to create a command-line interface.

<br>

## Command-Line Interface
The script provides a command-line interface with several options:

`project_path`: The path to the project directory. The default value is "example".

`delete_existing`: A boolean flag that indicates whether to delete existing files in the project directory. The default value is `False`.

`model`: The name of the AI model to use. The default value is "gpt-4".

`temperature`: The temperature parameter for the AI model, which controls the randomness of the model's output. The default value is `0.1`.

`steps_config`: The configuration of steps to run. The default value is "default".

`verbose`: A boolean flag that controls the verbosity of the logging. If `True`, the logging level is set to `DEBUG`. Otherwise, the logging level is set to `INFO`. The default value is `False`.

`run_prefix`: A prefix for the run, which can be used if you want to run multiple variants of the same project and later compare them. The default value is an empty string.

<br>

## Usage
To run the script, you can use the following command:

```
bash
python gpt_engineer/main.py --project_path [path] --delete_existing [True/False] --model [model] --temperature [temperature] --steps [steps_config] --verbose [True/False] --run_prefix [prefix]
```

You can replace the placeholders with the appropriate values. For example, to run the script with the default configuration on a project in the "my_project" directory, you can use the following command:

```
bash
python gpt_engineer/main.py --project_path my_project
```

<br>

## Conclusion
The `main.py` script provides a flexible and user-friendly interface for running the GPT-Engineer system. It allows you to easily configure the AI model, the steps to run, and other parameters. The script also provides detailed logging, which can be useful for debugging and understanding the behavior of the system.
