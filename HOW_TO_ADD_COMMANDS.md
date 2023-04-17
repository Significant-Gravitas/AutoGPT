# Adding New Commands to AutoGPT

In order to add new commands to AutoGPT you need to modify a couple of files. 

A command is a tuple `(str, str, dict)` and specifically:

```
("my command description", "my_command_name", {"param1": "<param1>", "param2": "<param2>", })
```

this tuple needs to be appended to the `commands` list in `prompt.py` in a proper place (start at line 43) before the `prompt_generator.add_command` loop:

```
commands.append(
    (
        "my command description",
        "my_command",
        {"param1": "<param1>", "param2": "<param2>", }
    ),
)
```

Now you need to update `app.py` in the `execute_command` function adding a dedicated `elif` statement where you will call the actual implementation of your command:

```
...
elif command_name == "my_command":
    return my_command(arguments["param1"], arguments["param1"])
...
```

the function `my_command` should return a string and needs to be properly located in the `commands` package and imported in `app.py`.

## Special Commands

If your command may interact with the system (e.g. creating files, downloading data...) you may want to ask specific permission to the user.

In order to do this you need to change the script arguments updating `args.py` adding:

```
...
parser.add_argument(
    '--allow-my-command',
    action='store_true',
    dest='allow_my_command',
    help='Describe why you need special permissions'
)
...
```

Now you need to change also `prompt.py` checking the current configuration:

```
if cfg.allow_my_command:
    commands.append(
        (
            "my command description",
            "my_command",
            {"param1": "<param1>", "param2": "<param2>", }
        ),
    )
```
