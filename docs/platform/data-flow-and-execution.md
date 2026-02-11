# Data Flow & Execution

## Overview

Understanding how agents execute is key to building effective workflows. This guide explains how data flows through an agent, what determines execution order, and how to work with lists and errors.

## Execution Order

Agent execution is entirely **determined by data flow**. There is no separate execution flow or ordering mechanism — data dependencies are the only thing that controls which block runs when.

### How It Works

1. **Execution starts from input blocks**, which yield their data when the agent is triggered (either manually or via a trigger/schedule)
2. The next block to run is whichever block has **all of its connected inputs satisfied**
3. This continues block by block, following the data flow, until all blocks have executed or an unhandled error occurs
4. **Output blocks** collect the final results and present them to the user

### Required Inputs

A block will only execute when:

- All **connected input pins** have received data from their upstream blocks
- All **required input pins** have values — either from a connection or from a hardcoded value set directly on the block

This means you can have blocks that don't depend on each other execute in any order, while blocks that depend on the output of another block will always wait.

## Working with Pins

### Pin Types

Input and output pins are typed. Common types include:

- **Text**: String values
- **Number**: Numeric values
- **File**: File uploads or downloads
- **List**: Arrays of items
- **Boolean**: True/false values
- **Object**: Structured data

Connections can only be made between compatible pin types.

### Data Flow Visualisation

When an agent is running, you can see data moving through the workflow in real time. Data flow is represented by a **coloured bead** that slides along each connection line from the output pin to the input pin, giving you a clear visual of what's happening.

## Working with Lists

Blocks can handle list data in flexible ways:

- **Outputting lists**: Some blocks produce a list of items as their output. You can choose to receive the full list as a single output or receive individual items one at a time.
- **Iterating over lists**: You can send a list into a block that iterates through its contents, yielding each item one by one. This is useful for processing each item in a list independently.

This makes it straightforward to build agents that process batches of data — for example, fetching a list of URLs and then processing each one through an AI block.

## Error Handling

When a block fails during execution, it does **not** automatically stop the entire agent. Instead:

1. The failed block produces data on its **error pin**
2. What happens next depends on how you've wired the agent

### Handling Errors Gracefully

You have full control over error handling through the block connections:

- **Surface the error**: Connect the error pin to an output block to return the error as part of the agent's result. This is useful for debugging or when you want users to see what went wrong.
- **Handle and continue**: Connect the error pin to other blocks that provide fallback behaviour. For example, retry with different settings, use a default value, or route to an alternative workflow path.
- **Ignore the error**: If the error pin is not connected, the error data is simply not propagated. Downstream blocks that depend on the failed block's normal output pins will not execute (since their inputs won't be satisfied).

{% hint style="info" %}
Building robust agents means thinking about what happens when things go wrong. Consider connecting error pins to output blocks during development so you can see any issues, then add proper error handling once your agent is working.
{% endhint %}

## Execution Summary

| Concept | How It Works |
|---------|-------------|
| **Execution order** | Determined entirely by data flow — blocks run when all inputs are ready |
| **Starting point** | Input blocks yield data first |
| **Ending point** | Output blocks collect final results |
| **Parallel execution** | Blocks with no dependencies on each other can execute in any order |
| **Error handling** | Failed blocks yield data on their error pin — you decide what to do with it |
| **Lists** | Can be processed as a whole or iterated item by item |
| **Visual feedback** | Coloured beads slide along connection lines during execution |
