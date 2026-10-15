"""AutoPilot's shared model gateway: one Anthropic Messages front, three upstreams.

The Claude Code CLI only speaks Anthropic's Messages API. This package is the
stateless loopback front it talks to on every non-Anthropic route, and the
adapters that carry each request to the real upstream: a pass-through for
servers that already speak ``/v1/messages``, OpenAI Chat Completions, and
ChatGPT's Codex Responses backend. Every request is translated whole, so the
gateway keeps no conversation state between them.
"""
