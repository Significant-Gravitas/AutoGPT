## Function `prompt_user()`

This function is responsible for prompting the user to set up an AI-Assistant. It takes no input parameters, and returns an instance of the `AIConfig` class that contains the user's input.

#### Example usage:
```python
ai_config = prompt_user()
```

#### Returns:
- `AIConfig`: An instance of the AIConfig class containing the user's input

#### Example output:
```
Welcome to Auto-GPT! [32mrun with '--help' for more information.[0m
Create an AI-Assistant: [32mEnter the name of your AI and its role below. Entering nothing will load defaults.[0m
Name your AI: [32mFor example, 'Entrepreneur-GPT'[0m
AI Name: 
Entrepreneur-GPT here! [94mI am at your service.[0m
Describe your AI's role: [32mFor example, 'an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.'[0m
Entrepreneur-GPT is: 
Enter up to 5 goals for your AI: [32mFor example: 
Increase net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'[0m
Enter nothing to load defaults, enter nothing when finished.
[94mGoal 1: [0m
```