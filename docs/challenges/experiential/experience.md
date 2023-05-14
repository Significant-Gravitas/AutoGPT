# Applying past Experience

## Status
NOTE: This is a draft and may need be reviewed/revised a few times before it's making sense. So please feel free editing this as needed!

## Agents need to be able to apply past experience

So we should add a benchmark to see how good an agent is at applying experience to future tasks.

For instance, imagine it editing/updating a file successfully and 5 minutes later it wants to use interactive editors like nano, vim etc - that's a recurring topic on GitHub.

We need to modify agent.py to reinforce steps (actions) that worked (evaluating the system message), versus those that didn't work.
This may involve recursively calling the LLM/GPT to evaluate the state (success/error/warning) of an operation (action/command).

A simple test case would be telling it to update a file with some hand holding, and afterwards leaving out the hand holding and counting how many times it succeeds or not (e.g. 3/10 attempts: 30%).

Being able to apply past experience is an essential part of learning. This would be a good starting point.


## Scope

Typically, an agent has several options to arrive at its goal.
Imagine the task being to retrieve some file over http (browse, download, python, shell)


However, some should be preferred over others (think fitness/reward function).

In other words, we need to look at the agent request/response loop and determine which metrics we could use to encode what works better in comparison to other options:

- errors of an operation
- warnings
- duration
- number of steps taken (!)
- number of API tokens required
- ...


## Success Evaluation

Success will be measured by the percentage of tests that an agent passes, re-running the same task without alternating/switching its tools in a manner that would add to the costs of the operation (again, think utility function).

Imagine we are telling the agent to update/edit 100 files from 1..100 inside the workspace.
If it comes up with a script/program to do so, the outcome will be deterministic - but if it's using the request/response mechanism, it needs to stay focused on using tools that previously worked (to provide this experience, the first instruction could be edit file using sed/awk or the using the new update file command)

