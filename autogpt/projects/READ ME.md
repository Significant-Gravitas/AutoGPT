

# Agent, Project and ProjectsBroker Classes

This module provides classes to manage projects and their related settings, including the organization of multiple projects, agents within those projects, and project budgets.

## Table of Contents

1. [Classes and Their Relations](#classes-and-their-relations)
2. [Subfolder Structure and Backup Folder](#subfolder-structure-and-backup-folder)
3. [Working with projects](#working-with-projects)

## Classes and Their Relations

The main class in this module is:

- `AgentModel`: Represents the configuration settings for an AI agent.
- `Project`: Represents an AI project, including its agents, budget, and other related information.
- `ProjectsBroker`: Manages multiple AI projects, including creating new projects and retrieving existing projects.

This module provides a convenient way to manage and configure AI agent settings, including loading and saving agent configurations, and handling agent attributes such as goals, roles, and model information.

The `AgentModel` class has several attributes and methods for handling agent configurations. Some of its main attributes include:
- agent_name
- agent_goals
- agent_role
- agent_model (optional)
- agent_model_type (optional)
- prompt_generator (optional)
- command_registry (optional)

The methods in the `AgentModel` class help manage agent configurations:

- `__init__()`: Initializes the `AgentModel` instance with the given attributes.
- `load_agent()`: Loads agent data from a dictionary and returns an `AgentModel` instance.
- `save()`: Saves the `AgentModel` object as a dictionary representation.
- `load()`: DEPRECATED method to maintain backward compatibility. Returns an `AgentModel` instance with parameters loaded from a YAML file if it exists, otherwise returns an instance with no parameters.
- `to_dict()`: Returns a dictionary representation of the `AgentModel` object.

The Project class has several attributes and methods for handling project configurations. Some of its main attributes include:

- project_name
- project_budget
- agent_team
- version

The methods in the Project class help manage project configurations:
- `__init__()`: Initializes the Project instance with the given attributes.
- `to_dict()`: Returns a dictionary representation of the Project object.

The ProjectsBroker class has the following methods:
- `__init__()`: Initializes the ProjectsBroker instance.
- `get_projects()`: Retrieves a list of all available projects.
- `create_project()`: Creates a new project and adds it to the list of projects.

## Subfolder Structure and Backup Folder

The subfolder structure of the module is organized as follows:

```
autogpt
├── projects
│   ├── project_1
│   │   ├── settings.yaml
│   ├── project_2
│   │   ├── settings.yaml
│   └── ...
```

Each project has its own folder (project_1, project_2, etc.) within the projects directory. Inside each project folder, there are individual folders for each agent (agent_1, agent_2, etc.).

When a new project is created, a backup folder is also created to store previous versions of the project's configurations. This allows for easy recovery or sharing.


## Working with multiple projects

You must remove these two lines from project.py if you intend to change coreto manage multiple projects
```python
shutil.rmtree(current_project_foldername)
createdir = True  
```
