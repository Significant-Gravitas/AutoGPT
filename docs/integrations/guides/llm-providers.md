# LLM Providers

There are several providers that AutoGPT users can use for running inference with LLM models.

## Llama API

Llama API is a Meta-hosted API service that helps you integrate Llama models quickly and efficiently. Using OpenAI compatibility endpoints, you can easily access the power of Llama models without the need for complex setup or configuration!

Join the [waitlist](https://llama.developer.meta.com/?utm_source=partner-autogpt&utm_medium=readme) to get access!

Try the Llama API provider by selecting any of the following LLM Model names from the AI blocks:

* Llama-4-Scout-17B-16E-Instruct-FP8
* Llama-4-Maverick-17B-128E-Instruct-FP8
* Llama-3.3-8B-Instruct
* Llama-3-70B-Instruct

## MiniMax

[MiniMax](https://www.minimaxi.com/) is an AI technology company that offers powerful foundation models. MiniMax provides an OpenAI-compatible API, making it easy to integrate their models into AutoGPT.

Get your API key from the [MiniMax Platform](https://platform.minimaxi.com/).

Try the MiniMax provider by selecting any of the following LLM Model names from the AI blocks:

* MiniMax-M2.5
* MiniMax-M2.5-highspeed

### How It Works

When an AI block is configured with a MiniMax model, AutoGPT routes requests through the OpenAI-compatible MiniMax endpoint (`https://api.minimax.io/v1`) using your configured provider credentials. The block sends the prompt/messages payload and enforces the selected response mode, including standard text, JSON-object output, and tool-calling when enabled.

The block also applies normal runtime safeguards before sending the request: prompt compression for long inputs, output token budgeting against model limits, and explicit empty-response checks. If the provider returns tool calls, AutoGPT maps them into the platform's internal tool-call structure so downstream blocks can execute them consistently.

### Use Case

**Structured extraction at scale:** Convert long documents into validated JSON outputs for downstream automation.

**Agentic workflows with tools:** Let the model choose and invoke tools while preserving consistent tool-call handling in AI blocks.

**Long-context synthesis:** Analyze large multi-message contexts and generate concise summaries or decisions in one pass.
