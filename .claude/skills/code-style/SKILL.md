---
name: code-style
description: Python code style preferences for the AutoGPT backend. Apply when writing or reviewing Python code. TRIGGER when writing new Python code, reviewing PRs, or refactoring backend code.
user-invocable: false
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Code Style

## Imports

- **Top-level only** — no local/inner imports. Move all imports to the top of the file.

## Typing

- **No duck typing** — avoid `hasattr`, `getattr`, `isinstance` for type dispatch. Use proper typed interfaces, unions, or protocols.
- **Pydantic models** over dataclass, namedtuple, or raw dict for structured data.
- **No linter suppressors** — avoid `# type: ignore`, `# noqa`, `# pyright: ignore` etc. 99% of the time the right fix is fixing the type/code, not silencing the tool.

## Code Structure

- **List comprehensions** over manual loop-and-append.
- **Early return** — guard clauses first, avoid deep nesting.
- **Flatten inline** — prefer short, concise expressions. Reduce `if/else` chains with direct returns or ternaries when readable.
- **Modular functions** — break complex logic into small, focused functions rather than long blocks with nested conditionals.

## Review Checklist

Before finishing, always ask:
- Can any function be split into smaller pieces?
- Is there unnecessary nesting that an early return would eliminate?
- Can any loop be a comprehension?
- Is there a simpler way to express this logic?
