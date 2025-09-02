#!/bin/bash

# This script sets up the FastAPI project with OpenAI streaming and tool calling demo.
# It installs dependencies, creates the necessary files, prompts for OpenAI API key,
# and runs the server. Once running, open http://localhost:8000 in your browser.

# Prompt for OpenAI API key
read -p "Enter your OpenAI API key: " api_key

# Install required packages
pip install fastapi uvicorn openai

# Create backend.py
cat << EOF > backend.py
# backend.py
# Install required packages: pip install fastapi uvicorn openai

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import openai
import asyncio
import json

app = FastAPI()

# Set your OpenAI API key
openai.api_key = "$api_key"  # Inserted from script

# Define example tools for demonstration
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

async def stream_openai_response(prompt: str):
    # State to accumulate tool calls since they stream in deltas
    tool_calls = []
    current_tool_call = None

    response = await asyncio.to_thread(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",  # Or "gpt-4" if available; note: use a model that supports tools
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="auto",
        stream=True
    )

    for chunk in response:
        delta = chunk.choices[0].delta

        # Handle regular content streaming
        if delta.get("content"):
            yield f"data: {delta.content}\n\n"

        # Handle tool call deltas
        if delta.get("tool_calls"):
            for tool_delta in delta.tool_calls:
                index = tool_delta.index if hasattr(tool_delta, 'index') else 0  # Handle indexing if multiple calls

                # Initialize if new tool call
                if len(tool_calls) <= index:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

                # Accumulate id, if present
                if tool_delta.id:
                    tool_calls[index]["id"] = (tool_calls[index]["id"] or "") + tool_delta.id

                # Accumulate function name
                if tool_delta.function and tool_delta.function.name:
                    tool_calls[index]["function"]["name"] = (
                        tool_calls[index]["function"]["name"] or ""
                    ) + tool_delta.function.name

                # Accumulate arguments (which come as JSON strings in chunks)
                if tool_delta.function and tool_delta.function.arguments:
                    tool_calls[index]["function"]["arguments"] = (
                        tool_calls[index]["function"]["arguments"] or ""
                    ) + tool_delta.function.arguments

        # Check if the chunk indicates completion (finish_reason)
        finish_reason = chunk.choices[0].finish_reason
        if finish_reason == "tool_calls" and tool_calls:
            # For demo, yield an HTML element representing the tool call(s)
            for tool_call in tool_calls:
                # Parse arguments if complete JSON
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                    args_str = json.dumps(args, indent=2)
                except json.JSONDecodeError:
                    args_str = tool_call["function"]["arguments"]  # Fallback if incomplete

                html = f'<div class="tool-call" style="border: 1px solid #ccc; padding: 10px; margin: 10px 0; background-color: #f9f9f9;">' \\
                       f'<strong>Tool Called:</strong> {tool_call["function"]["name"]}<br>' \\
                       f'<strong>Arguments:</strong><pre>{args_str}</pre>' \\
                       f'</div>'
                yield f"data: {html}\n\n"

            # Note: In a real implementation, you'd execute the tool here and append the result
            # to messages, then make another API call to continue the response.
            # For this simple demo, we just show the tool call HTML and stop.

    # Clear tool calls for next stream if needed (but since it's per request, not necessary)

@app.get("/stream-chat")
async def stream_chat(prompt: str):
    return StreamingResponse(stream_openai_response(prompt), media_type="text/event-stream")

# Serve the frontend HTML at root
frontend_content = """
<!-- frontend.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SSE OpenAI Stream Demo with Tools</title>
    <style>
        .tool-call {
            border: 1px solid #ccc;
            padding: 10px;
            margin: 10px 0;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>OpenAI Chat Stream Demo with Tool Calling</h1>
    <input type="text" id="prompt" placeholder="Enter your prompt (e.g., 'What\\'s the weather in Paris?')">
    <button onclick="startStream()">Send</button>
    <div id="response" style="white-space: pre-wrap;"></div>

    <script>
        function startStream() {
            const prompt = document.getElementById('prompt').value;
            const responseDiv = document.getElementById('response');
            responseDiv.innerHTML = '';  // Clear previous response

            const eventSource = new EventSource(\`/stream-chat?prompt=\${encodeURIComponent(prompt)}\`);

            eventSource.onmessage = function(event) {
                // Append the data as HTML to allow rendering of elements
                responseDiv.innerHTML += event.data;
            };

            eventSource.onerror = function(error) {
                console.error('EventSource failed:', error);
                eventSource.close();
            };
        }
    </script>
</body>
</html>
"""

@app.get("/", return_directly=True)
async def root():
    return HTMLResponse(frontend_content)

# Run with: uvicorn backend:app --reload
EOF

# Run the FastAPI app
echo "Starting the server... Open http://localhost:8000 in your browser to access the demo."
uvicorn backend:app --reload