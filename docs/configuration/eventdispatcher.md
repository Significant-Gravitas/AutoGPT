## Event Dispatcher

By default, Auto-GPT writes output to the prompt. The event dispatcher enables sending all console output to an endpoint where further actions, such as an enterprise service bus or 3rd party applications reacting to ongoing output, can take place. The event dispatcher asynchronously fires and forgets such events through the HTTP POST method.

To enable the event dispatcher, set these variables in your `.env`:

```bash
EVENT_DISPATCHER_ENABLED=True
EVENT_DISPATCHER_PROTOCOL=http
EVENT_DISPATCHER_HOST=localhost
EVENT_DISPATCHER_ENDPOINT=/events
EVENT_DISPATCHER_PORT=45000
```