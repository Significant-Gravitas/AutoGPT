# Configurator Module

This module contains a function `create_config` that updates the config object with the given arguments.

```python
def create_config(
        continuous: bool,
        continuous_limit: int,
        ai_settings_file: str,
        skip_reprompt: bool,
        speak: bool,
        debug: bool,
        gpt3only: bool,
        gpt4only: bool,
        memory_type: str,
        browser_name: str,
        allow_downloads: bool,
        skip_news: bool
) -> None:
```

### Parameters

- `continuous` (bool): Whether to run in continuous mode.
- `continuous_limit` (int): The number of times to run in continuous mode.
- `ai_settings_file` (str): The path to the ai_settings.yaml file.
- `skip_reprompt` (bool): Whether to skip the re-prompting messages at the beginning of the script.
- `speak` (bool): Whether to enable speak mode.
- `debug` (bool): Whether to enable debug mode.
- `gpt3only` (bool): Whether to enable GPT3.5 only mode.
- `gpt4only` (bool): Whether to enable GPT4 only mode.
- `memory_type` (str): The type of memory backend to use.
- `browser_name` (str): The name of the browser to use when using selenium to scrape the web.
- `allow_downloads` (bool): Whether to allow Auto-GPT to download files natively.
- `skip_news` (bool): Whether to suppress the output of latest news on startup.

### Returns

- None

#### Remarks

- This function updates the global `CFG` object.
- Some of the arguments are used to set flags in the global `CFG` object.
- If the `ai_settings_file` parameter is provided, it is used to validate the YAML file passed through.
- If the YAML file validation fails, an error message is printed and the script is exited.
- The `memory_type` parameter is used to update the memory backend the global `CFG` object uses. It first checks if the type is supported or not. If it is not supported, it prints out a warning message and sets the `CFG.memory_backend` to the default memory backend.
- If `allow_downloads` is set to True, it prints out a warning message that stresses the importance of monitoring downloaded files and never opening a file if unsure of its origin. It then sets the `CFG.allow_downloads` to True. 
- Any other arguments that are set to True will result in a log message being printed to the console.