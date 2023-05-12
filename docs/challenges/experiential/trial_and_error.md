# Trial and Error

The challenge focuses on designing an AI agent that learns from trial and error by recognizing invariants and exploring variable factors. The agent's objective is to achieve a specific goal by executing actions with different parameter combinations. It maintains a memory of successful and unsuccessful experiences, prioritizing previously successful actions and experimenting with variable factors to improve performance. The challenge evaluates the agent's ability to adapt, identify invariants, explore variable factors, and converge towards more effective strategies for accomplishing the goal.


## Description

In this example, the AI agent needs to understand and differentiate between the invariant and variable parts of a CLI command. The command uses the grep tool to search for a given string inside a configurable file using a regular expression (regex). The invariant part of the command is the grep -r portion, which remains constant. The variable parts are the "search_string" and "/path/to/configurable_file", which can vary depending on the specific search string and file path provided by the user. By recognizing the invariants and variable factors, the agent can adapt its behavior to execute the command with different search strings and file paths effectively.

## Scope

The challenge focuses on designing an AI agent capable of memorizing invariant and variable parts of its actions and utilizing this knowledge for future tasks. The agent should recognize patterns within its actions, distinguishing between invariant and variable parts. It should create an efficient memory structure to store and retrieve information about these parts. By leveraging its memory, the agent can make adaptive decisions based on past experiences, applying successful strategies involving invariant knowledge while experimenting with variable parts. 



## Success Evaluation

The success of the challenge is determined by the agent's ability to recognize, store, and utilize invariant and variable parts effectively, demonstrating adaptive behavior and improved performance across different tasks.

## Future Challenges
In this advanced variation of the challenge, participants design an AI agent capable of managing multiple invariant/variable tuples per action, handling diverse CLI tools or actions/commands, and associating invariant/variable tuples between subsequent pairs of actions/commands. The agent aims to recognize, store, and recall various sets of invariant/variable tuples along with their associated actions. By leveraging this memory, the agent makes adaptive decisions, optimizes performance, and identifies correlations between different actions/commands. Success is measured by the agent's ability to handle multiple tuples, adapt to different tools/commands, and effectively associate invariant/variable tuples across subsequent actions/commands, showcasing advanced adaptability and memory management skills.

