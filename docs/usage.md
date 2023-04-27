# Usage

1. Run the `autogpt` Python module in your terminal.
* On Linux/MacOS:
   ```   
   ./run.sh
   ```
* On Windows:
   ```   
   .\run.bat
   ```
   Running with `--help` after `.\run.bat` lists all the possible command line arguments you can pass.

2. After each response from Auto-GPT, choose from the options to authorize command(s),
exit the program, or provide feedback to the AI.
   1. Authorize a single command by entering `y`
   2. Authorize a series of _N_ continuous commands by entering `y -N`. For example, entering `y -10` would run 10 automatic iterations.
   3. Enter any free text to give feedback to Auto-GPT.
   4. Exit the program by entering `n`


## Command Line Arguments
Here are some common arguments you can use when running Auto-GPT:
> Replace anything in angled brackets (<>) to a value you want to specify

* View all available command line arguments
    ```    
    python -m autogpt --help
    ```
* Run Auto-GPT with a different AI Settings file
    ```    
    python -m autogpt --ai-settings <filename>
    ```
* Specify a memory backend
    ```    
    python -m autogpt --use-memory  <memory-backend>
    ```

> **NOTE**: There are shorthands for some of these flags, for example `-m` for `--use-memory`. Use `python -m autogpt --help` for more information

### Speak Mode 

Enter this command to use TTS _(Text-to-Speech)_ for Auto-GPT

```
python -m autogpt --speak
```

### 💀 Continuous Mode ⚠️

Run the AI **without** user authorization, 100% automated.
Continuous mode is NOT recommended.
It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorize.
Use at your own risk.

1. Run the `autogpt` python module in your terminal:

    ```    
    python -m autogpt --continuous
    ```

2. To exit the program, press Ctrl + C

### GPT3.5 ONLY Mode

If you don't have access to the GPT4 api, this mode will allow you to use Auto-GPT!

``` shell
python -m autogpt --gpt3only
```

### GPT4 ONLY Mode

If you do have access to the GPT4 api, this mode will allow you to use Auto-GPT solely using the GPT-4 API for increased intelligence (and cost!)

``` shell
python -m autogpt --gpt4only
```

## Logs

Activity and error logs are located in the `./output/logs`

To print out debug logs:

``` shell
python -m autogpt --debug
```