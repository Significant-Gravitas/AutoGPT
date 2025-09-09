# backend.py
# Install required packages: pip install fastapi uvicorn openai

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

# Define example tools for demonstration - proper OpenAI format
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


# Mock function to execute tools
def execute_tool(tool_name: str, parameters: dict) -> str:
    """Execute a tool and return the result"""
    if tool_name == "get_current_weather":
        location = parameters.get("location", "Unknown")
        unit = parameters.get("unit", "fahrenheit")
        # Mock weather data
        temp = 72 if unit == "fahrenheit" else 22
        return f"The weather in {location} is currently {temp}° {unit.upper()} with partly cloudy skies."
    return "Tool not found"


async def stream_openai_response(prompt: str):
    """Stream OpenAI responses with proper tool calling support"""
    # Build initial messages list
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    logger.info(f"Received prompt: {prompt}")

    # Loop to handle tool calls and continue conversation
    while True:
        try:
            logger.info("Creating OpenAI chat completion stream...")
            logger.info(f"Current messages: {json.dumps(messages, indent=2)}")

            # Use chat.completions API (standard OpenAI format)
            stream = client.chat.completions.create(
                model="gpt-4",  # Use a model that supports tools
                messages=messages,  # type: ignore
                tools=tools,  # type: ignore
                tool_choice="auto",
                stream=True,
            )

            # Variables to accumulate the response
            assistant_message = ""
            tool_calls = []
            finish_reason = None

            # Process the stream
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    # Capture finish reason
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        finish_reason = choice.finish_reason
                        logger.info(f"Finish reason: {finish_reason}")

                    # Handle content streaming
                    if hasattr(delta, "content") and delta.content:
                        assistant_message += delta.content
                        # Stream word by word for nice effect
                        words = delta.content.split(" ")
                        for word in words:
                            if word:  # Skip empty strings
                                yield f"data: {json.dumps({'type': 'text', 'content': word + ' '})}\n\n"
                                await asyncio.sleep(0.02)

                    # Handle tool calls - accumulate them properly
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index

                            # Ensure we have a tool call object at this index
                            while len(tool_calls) <= idx:
                                tool_calls.append(
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )

                            # Accumulate the tool call data
                            if tc_chunk.id:
                                tool_calls[idx]["id"] += tc_chunk.id
                            if tc_chunk.function:
                                if tc_chunk.function.name:
                                    tool_calls[idx]["function"][
                                        "name"
                                    ] += tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    tool_calls[idx]["function"][
                                        "arguments"
                                    ] += tc_chunk.function.arguments

            logger.info(f"Stream complete. Finish reason: {finish_reason}")
            logger.info(f"Tool calls: {tool_calls}")

            # Check if we need to execute tools
            if finish_reason == "tool_calls" and tool_calls:
                logger.info(f"Processing {len(tool_calls)} tool call(s)")

                # Add assistant message with tool calls to conversation
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message if assistant_message else None,
                        "tool_calls": tool_calls,
                    }
                )

                # Process each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_id = tool_call["id"]

                    # Parse arguments
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}

                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                    # Show tool call UI
                    html = f"""<div class="tool-call-container" style="margin: 20px 0; animation: slideIn 0.3s ease-out;">
                        <div class="tool-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; border-radius: 8px 8px 0 0; font-weight: bold; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">🔧</span>
                            <span>Calling Tool: {tool_name}</span>
                        </div>
                        <div class="tool-body" style="background: #f7f9fc; padding: 15px; border: 1px solid #e1e4e8; border-top: none; border-radius: 0 0 8px 8px;">
                            <div style="color: #586069; font-size: 12px; margin-bottom: 8px;">Parameters:</div>
                            <pre style="background: white; padding: 10px; border-radius: 4px; border: 1px solid #e1e4e8; margin: 0; font-size: 13px; color: #24292e;">{json.dumps(tool_args, indent=2)}</pre>
                        </div>
                    </div>"""
                    yield f"data: {json.dumps({'type': 'html', 'content': html})}\n\n"
                    await asyncio.sleep(0.3)

                    # Execute the tool
                    executing_html = f"""<div class="tool-executing" style="margin: 10px 0; padding: 10px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; color: #856404;">
                        <span style="animation: pulse 1.5s infinite;">⏳</span> Executing {tool_name}...
                    </div>"""
                    yield f"data: {json.dumps({'type': 'html', 'content': executing_html})}\n\n"
                    await asyncio.sleep(0.5)

                    # Get tool result
                    tool_result = execute_tool(tool_name, tool_args)
                    logger.info(f"Tool result: {tool_result}")

                    # Show result
                    result_html = f"""<div class="tool-result" style="margin: 10px 0; padding: 15px; background: #e8f5e9; border: 1px solid #4caf50; border-radius: 6px;">
                        <div style="color: #2e7d32; font-weight: bold; margin-bottom: 8px;">📊 Tool Result:</div>
                        <div style="color: #1b5e20;">{tool_result}</div>
                    </div>"""
                    yield f"data: {json.dumps({'type': 'html', 'content': result_html})}\n\n"

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": tool_result,
                        }
                    )

                # Show processing message
                processing_html = """<div style="margin: 15px 0; padding: 10px; background: #e3f2fd; border: 1px solid #2196f3; border-radius: 6px; color: #1565c0;">🤖 Processing tool results...</div>"""
                yield f"data: {json.dumps({'type': 'html', 'content': processing_html})}\n\n"
                await asyncio.sleep(0.5)

                # Continue the loop to get final response
                logger.info("Making follow-up call with tool results...")
                continue
            else:
                # No tool calls, conversation complete
                logger.info("Conversation complete")
                break

        except Exception as e:
            logger.error(f"Error in stream: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'text', 'content': f'Error: {str(e)}'})}\n\n"
            break


async def stream_openai_response_old(prompt: str):
    # Build messages list
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_calls_to_execute = []

    logger.info(f"Received prompt: {prompt}")

    try:
        logger.info("Creating OpenAI response stream...")
        response = client.responses.create(  # type: ignore
            model="gpt-5-nano-2025-08-07",  # Latest model
            input=messages,  # type: ignore
            tools=tools,  # type: ignore
            tool_choice="auto",
            stream=True,
        )
        logger.info("Response stream created successfully")

        chunk_count = 0
        assistant_message = ""
        for chunk in response:
            chunk_count += 1
            logger.info(
                f"Processing chunk #{chunk_count}, type: {type(chunk).__name__}"
            )
            logger.info(f"Chunk attributes: {dir(chunk)}")

            # The new API returns ResponseCreatedEvent objects
            # Extract the content directly from the chunk
            if hasattr(chunk, "output"):
                logger.info(f"Chunk has output: {chunk.output}")
                # Handle text output
                if chunk.output:
                    for output_item in chunk.output:
                        if hasattr(output_item, "content") and output_item.content:
                            logger.info(
                                f"Yielding content: {output_item.content[:50]}..."
                            )
                            # Stream word by word for better visual effect
                            words = output_item.content.split(" ")
                            for word in words:
                                yield f"data: {json.dumps({'type': 'text', 'content': word + ' '})}\n\n"
                                await asyncio.sleep(0.02)  # Small delay between words
                        elif hasattr(output_item, "text") and output_item.text:
                            logger.info(f"Yielding text: {output_item.text[:50]}...")
                            assistant_message += output_item.text
                            # Stream word by word
                            words = output_item.text.split(" ")
                            for word in words:
                                yield f"data: {json.dumps({'type': 'text', 'content': word + ' '})}\n\n"
                                await asyncio.sleep(0.02)

            # Alternative: If the response is just text
            if hasattr(chunk, "text") and chunk.text:
                logger.info(f"Chunk has text: {chunk.text[:50]}...")
                assistant_message += chunk.text
                # Stream character by character for visual effect
                for char in chunk.text:
                    yield f"data: {json.dumps({'type': 'text', 'content': char})}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for visual streaming effect

            # Handle tool calls if present
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                logger.info(f"Chunk has tool_calls: {chunk.tool_calls}")

                # Process each tool call
                for tool_call in chunk.tool_calls:
                    tool_name = getattr(tool_call, "name", "Unknown Tool")
                    tool_params = getattr(tool_call, "parameters", {})
                    tool_id = getattr(tool_call, "id", f"tool_{chunk_count}")

                    # Store tool call for execution
                    tool_calls_to_execute.append(
                        {"id": tool_id, "name": tool_name, "parameters": tool_params}
                    )

                    logger.info(f"Tool call: {tool_name} with params: {tool_params}")

                    # Create animated tool call UI
                    html = f"""<div class="tool-call-container" style="margin: 20px 0; animation: slideIn 0.3s ease-out;">
                        <div class="tool-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; border-radius: 8px 8px 0 0; font-weight: bold; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">🔧</span>
                            <span>Calling Tool: {tool_name}</span>
                        </div>
                        <div class="tool-body" style="background: #f7f9fc; padding: 15px; border: 1px solid #e1e4e8; border-top: none; border-radius: 0 0 8px 8px;">
                            <div style="color: #586069; font-size: 12px; margin-bottom: 8px;">Parameters:</div>
                            <pre style="background: white; padding: 10px; border-radius: 4px; border: 1px solid #e1e4e8; margin: 0; font-size: 13px; color: #24292e;">{json.dumps(tool_params, indent=2)}</pre>
                        </div>
                    </div>"""

                    yield f"data: {json.dumps({'type': 'html', 'content': html})}\n\n"
                    await asyncio.sleep(0.3)

                    # Execute the tool
                    executing_html = f"""<div class="tool-executing" style="margin: 10px 0; padding: 10px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; color: #856404;">
                        <span style="animation: pulse 1.5s infinite;">⏳</span> Executing {tool_name}...
                    </div>"""
                    yield f"data: {json.dumps({'type': 'html', 'content': executing_html})}\n\n"

                    # Actually execute the tool
                    tool_result = execute_tool(tool_name, tool_params)
                    logger.info(f"Tool result: {tool_result}")

                    # Show result
                    result_html = f"""<div class="tool-result" style="margin: 10px 0; padding: 15px; background: #e8f5e9; border: 1px solid #4caf50; border-radius: 6px;">
                        <div style="color: #2e7d32; font-weight: bold; margin-bottom: 8px;">📊 Tool Result:</div>
                        <div style="color: #1b5e20;">{tool_result}</div>
                    </div>"""
                    yield f"data: {json.dumps({'type': 'html', 'content': result_html})}\n\n"
                    await asyncio.sleep(0.5)

        logger.info(f"Finished processing {chunk_count} chunks")

        # If we had tool calls, make a second API call with the results
        if tool_calls_to_execute:
            logger.info("Making second API call with tool results...")

            # Append assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message if assistant_message else None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["parameters"]),
                            },
                        }
                        for tc in tool_calls_to_execute
                    ],
                }
            )

            # Append tool results
            for tool_call in tool_calls_to_execute:
                tool_result = execute_tool(tool_call["name"], tool_call["parameters"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result,
                    }
                )

            # Make second API call
            processing_html = """<div style="margin: 15px 0; padding: 10px; background: #e3f2fd; border: 1px solid #2196f3; border-radius: 6px; color: #1565c0;">🤖 Processing tool results...</div>"""
            yield f"data: {json.dumps({'type': 'html', 'content': processing_html})}\n\n"

            second_response = client.responses.create(  # type: ignore
                model="gpt-5-nano-2025-08-07", input=messages, stream=True  # type: ignore
            )

            # Stream the final response
            for chunk in second_response:
                if hasattr(chunk, "text") and chunk.text:
                    # Stream word by word
                    words = chunk.text.split(" ")
                    for word in words:
                        yield f"data: {json.dumps({'type': 'text', 'content': word + ' '})}\n\n"
                        await asyncio.sleep(0.02)
                elif hasattr(chunk, "output") and chunk.output:
                    for output_item in chunk.output:
                        if hasattr(output_item, "content") and output_item.content:
                            words = output_item.content.split(" ")
                            for word in words:
                                yield f"data: {json.dumps({'type': 'text', 'content': word + ' '})}\n\n"
                                await asyncio.sleep(0.02)
                        elif hasattr(output_item, "text") and output_item.text:
                            words = output_item.text.split(" ")
                            for word in words:
                                yield f"data: {json.dumps({'type': 'text', 'content': word + ' '})}\n\n"
                                await asyncio.sleep(0.02)

    except Exception as e:
        logger.error(f"Error in stream_openai_response: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'text', 'content': f'Error: {str(e)}'})}\n\n"


@app.get("/stream-chat")
async def stream_chat(prompt: str):
    logger.info(f"Stream chat endpoint called with prompt: {prompt}")
    return StreamingResponse(
        stream_openai_response(prompt), media_type="text/event-stream"
    )


# Serve the frontend HTML at root
frontend_content = """
<!-- frontend.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SSE OpenAI Stream Demo with Tools</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f7fa;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        #prompt {
            width: 70%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        button {
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 10px;
        }
        
        button:hover {
            opacity: 0.9;
        }
        
        #response {
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 100px;
            white-space: pre-wrap;
            line-height: 1.6;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
    </style>
</head>
<body>
    <h1>OpenAI Chat Stream Demo with Tool Calling</h1>
    <input type="text" id="prompt" placeholder="Enter your prompt (e.g., 'What\'s the weather in Paris?')">
    <button onclick="startStream()">Send</button>
    <div id="response" style="white-space: pre-wrap;"></div>

    <script>
        function startStream() {
            const prompt = document.getElementById('prompt').value;
            const responseDiv = document.getElementById('response');
            responseDiv.innerHTML = '';  // Clear previous response

            const eventSource = new EventSource(`/stream-chat?prompt=${encodeURIComponent(prompt)}`);
            
            // Add loading indicator
            responseDiv.innerHTML = 'Connecting to OpenAI...';
            
            eventSource.onopen = function() {
                responseDiv.innerHTML = '';  // Clear loading message
            };

            eventSource.onmessage = function(event) {
                try {
                    const parsed = JSON.parse(event.data);
                    if (parsed.type === 'html') {
                        // Tool call HTML
                        responseDiv.innerHTML += parsed.content;
                    } else if (parsed.type === 'text') {
                        // Regular text - append character/word incrementally
                        responseDiv.appendChild(document.createTextNode(parsed.content));
                    }
                } catch (e) {
                    // Fallback for non-JSON data
                    responseDiv.appendChild(document.createTextNode(event.data));
                }
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


@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return HTMLResponse(frontend_content)


# Run with: uvicorn backend:app --reload

if __name__ == "__main__":
    import socket

    import uvicorn

    # Use port 0 to get a random available port
    config = uvicorn.Config(app, host="0.0.0.0", port=0)
    server = uvicorn.Server(config)

    # Get the actual port after binding
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    _, port = sock.getsockname()
    sock.close()

    print(f"\n🚀 Server starting on http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
