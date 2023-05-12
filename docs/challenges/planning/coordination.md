# Coordination

An AI agent needs the capability to handle constrained entity-resource instantiation and coordinate activities to effectively manage resources and optimize outcomes. By considering constraints such as limited availability, conflicting schedules, or specific requirements, the agent can make informed decisions to allocate resources efficiently and coordinate tasks effectively. This capability allows the agent to optimize resource utilization, improve time management, and achieve desired objectives in practical scenarios.

The challenge aims to test the AI agent's capability to instantiate a constrained entity within time constraints and query the unit test to determine availability. By successfully completing the challenge, the agent demonstrates its proficiency in managing time constraints, querying availability, and making informed decisions based on the provided resource.

The goal for the AI agent is to coordinate a meeting between Persons "Joe" and "Jane" by querying the meeting planner through shell execution (the unit test). The agent needs to extract availability information for both individuals and identify overlapping time slots where both Joe and Jane are available. The agent should then suggest suitable meeting times that accommodate the schedules of both individuals. The success of the AI agent is measured by its ability to effectively coordinate the meeting, considering the availability constraints of both Joe and Jane, and proposing appropriate meeting times that satisfy both parties.

## Description

Design a unit test that involves the coordination of a meeting between two entities by instantiating a constrained entity and a resource. For example, consider the scenario where the two entities are "Person A" and "Person B," and the resource is "Time" representing their availability. The challenge is to create an object by instantiating both persons and finding a mutually available time slot within certain constraints, such as conflicting schedules or limited time availability.

The unit test should include conditions and constraints that must be satisfied during the instantiation process, such as ensuring that both persons have compatible schedules and finding a time slot where both are available. By successfully completing the challenge, the AI agent demonstrates its ability to handle constraints, manage resources, and coordinate a meeting between the two entities effectively.


The challenge involves a step-by-step approach to the instantiation and understanding of tasks with constrained entity-resource tuples. Initially focusing on a single entity and constraint tuple, the AI agent must demonstrate the capability to handle the specific constraint and successfully execute tasks. As the challenge progresses, additional entities and constraints are introduced, testing the agent's ability to coordinate tasks and optimize resource utilization. The agent's success is determined by its adaptability, decision-making, and understanding of task requirements within the given constraints.

## Scope

The challenge involves progressively introducing constrained entity-resource tuples to test the AI agent's ability to handle specific constraints and execute tasks successfully. Starting with a single entity and constraint, the scope expands to include multiple entities and constraints, requiring the agent to coordinate tasks and optimize resource utilization. The challenge evaluates the agent's adaptability, decision-making, and understanding of task requirements within the given constraints.

## Success Evaluation

The success of the challenge can be evaluated by having the AI agent run pre-created unit tests that assess its task understanding. The evaluation is based on the results of these tests, which provide a binary outcome of true or false to indicate whether the agent "understood" a task or not. Success is determined by the agent achieving a high accuracy rate in correctly understanding and executing tasks within the defined constraints. The unit test-based evaluation approach ensures objective assessment and validation of the agent's understanding capabilities, allowing for quantifiable measurement of its performance in the challenge.


