## AIConfig Class

The `AIConfig` class is responsible for managing the configuration information for an AI.

### Attributes

- `ai_name` (str): The name of the AI.
- `ai_role` (str): The description of the AI's role.
- `ai_goals` (list): The list of objectives the AI is supposed to complete.

### Methods

#### \_\_init\_\_

Initialize a class instance.

##### Parameters

- `ai_name` (str): The name of the AI. DEFAULT: "".
- `ai_role` (str): The description of the AI's role. DEFAULT: "".
- `ai_goals` (list): The list of objectives the AI is supposed to complete. DEFAULT: [].

##### Returns

None.

#### load

Returns class object with parameters (`ai_name`, `ai_role`, and `ai_goals`) loaded from a yaml file if the file exists, otherwise returns the class with no parameters.

##### Parameters

- `config_file` (str): The path to the config yaml file. DEFAULT: "../ai_settings.yaml".

##### Returns

- `cls` (object): A instance of the given `cls` object.

#### save

Saves the class parameters to the specified yaml file path.

##### Parameters

- `config_file` (str): The path to the config yaml file. DEFAULT: "../ai_settings.yaml".

##### Returns

None.

#### construct_full_prompt

Returns a prompt string to the user with a formatted class information.

##### Parameters

None.

##### Returns

- `full_prompt` (str): A string containing the intitial prompt for the user including the `ai_name`, `ai_role` and `ai_goals`.