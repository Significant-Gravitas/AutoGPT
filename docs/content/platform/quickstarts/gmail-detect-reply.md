# Detect replies to a scheduling email

This quick-start flow demonstrates how to read an incoming message, capture its `threadId`, and then fetch the whole conversation to check for a reply.

1. **Gmail Read** — search for your initial scheduling email and output its `threadId`.
2. **Gmail Get Thread** — pass the captured `threadId` to retrieve the conversation.
3. **Condition Block** — inspect the thread messages and branch if a reply is found.
4. **Gmail Reply** — send a follow-up in the same thread if needed.

Use this pattern to keep email-based workflows organized without manual HTTP calls.
