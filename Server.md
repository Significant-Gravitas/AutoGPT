# Run Auto-GPT as an HTTP server

A fastapi app is lives in `scripts/app.py`. You can start it from the project root by running `uvicorn app:app --reload --app-dir=./scripts`. The app is configured to run on port 8000 by default.

You should already have uvicorn installed if you ran `pip install -r requirements.txt`.

## API

### Make an HTTP POST request to localhost:8000/start 

here's an example of the request body format:

```json
{
    "ai_name": "HelloBot",
    "ai_role": "An AI that says 'Hello, World!'",
    "ai_goals": [
        "Write your message in a file called 'message.txt'.",
        "Shut down."
    ]
}
```

It will reply with something like this:

```json
{
    "messages": [
        {
            "title": "HELLOBOT THOUGHTS:",
            "message": "As an AI that says 'Hello, World!', my goal is to write a message in a file called 'message.txt' and shutdown. Therefore, I will use the 'write_to_file' command to write my message to 'message.txt' and then use the 'task_complete' command to shutdown. I don't need any arguments for the 'task_complete' command, but for 'write_to_file' command, I will use the 'file' argument to specify 'message.txt' and the 'text' argument to specify my message."
        },
        {
            "title": "REASONING:",
            "message": "I have a clear goal to achieve, so I will use the commands that will help me achieve that goal with the least number of steps."
        },
        {
            "title": "PLAN:",
            "message": ""
        },
        {
            "title": "- ",
            "message": "Use the 'write_to_file' command to write 'Hello, World!' to 'message.txt'."
        },
        {
            "title": "- ",
            "message": "Use the 'task_complete' command to shutdown."
        },
        {
            "title": "CRITICISM:",
            "message": "I do not have any criticisms at this time."
        },
        {
            "title": "NEXT ACTION: ",
            "message": "COMMAND = write_to_file  ARGUMENTS = {'file': 'message.txt', 'text': 'Hello, World!'}"
        }
    ]
}
```

Make sure you check the response headers for a header called `chat_id`. You will need to include this in your next request to continue that chat. This is how the server keeps track of which chat it is responding to.

In the future this can be made more secure.

### Make an HTTP POST request to localhost:8000/chat

This is how you give Auto-GPT permission to run its next command. Make sure you include the `chat_id` request header. Here's an example of the request body format:
```json
{
    "message": "y"
}
```

The response will look something like this:
```json
{
    "messages": [
        {
            "title": "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
            "message": ""
        },
        {
            "title": "SYSTEM: ",
            "message": "Command write_to_file returned: File written to successfully."
        },
        {
            "title": "HELLOBOT THOUGHTS:",
            "message": "I will shutdown using the 'task_complete' command."
        },
        {
            "title": "REASONING:",
            "message": "This is the final step of my goal, and it ensures that I shutdown properly."
        },
        {
            "title": "PLAN:",
            "message": ""
        },
        {
            "title": "- ",
            "message": "Use 'task_complete' command with reason 'Goal achieved' to shutdown"
        },
        {
            "title": "CRITICISM:",
            "message": ""
        },
        {
            "title": "NEXT ACTION: ",
            "message": "COMMAND = task_complete  ARGUMENTS = {'reason': 'Goal achieved'}"
        }
    ]
}
```

In this example you would find the file written by Auto-GPT in `./auto_gpt_workspace/<chat_id>/message.txt`.