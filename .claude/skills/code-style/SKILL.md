---
name: code-style
description: Code style and quality rules for the AutoGPT platform. Covers Python backend AND TypeScript/React frontend. Auto-applied when writing, reviewing, or refactoring code. TRIGGER when writing new code, reviewing PRs, refactoring, or when code quality issues are found.
user-invocable: false
metadata:
  author: autogpt-team
  version: "2.0.0"
---

# Code Style

## Python (Backend)

### Imports
- **Top-level only** — no local/inner imports. Move all imports to the top of the file.
- Lazy imports only for heavy optional dependencies (e.g., `openpyxl`) with a clear comment.

### Typing
- **No duck typing** — avoid `hasattr`, `getattr`, `isinstance` for type dispatch. Use typed interfaces, unions, or protocols.
- **Pydantic models** over dataclass, namedtuple, or raw dict for structured data.
- **No linter suppressors** — no `# type: ignore`, `# noqa`, `# pyright: ignore`. Fix the type/code instead.

### Code structure
- **List comprehensions** over manual loop-and-append.
- **Early return** — guard clauses first, avoid deep nesting.
- **Flatten inline** — short expressions, reduce `if/else` chains with direct returns or ternaries.
- **Modular functions** — break complex logic into small, focused functions.

### Logging
- **Lazy `%s` format** — `logger.info("Processing %s items", count)` not `logger.info(f"Processing {count} items")`. This avoids string formatting when the log level is disabled.

### Error handling
- **Broad exception handling** — catch specific exceptions, not bare `except:` or `except Exception:` unless re-raising.
- **Sanitize error paths** — use `os.path.basename()` when including file paths in error messages to avoid leaking directory structure.
- **TOCTOU awareness** — avoid check-then-act patterns for file access and credit charging. Use atomic operations where possible.

### FastAPI specifics
- **`Security()` vs `Depends()`** — use `Security()` for auth dependencies to get proper OpenAPI security spec generation.
- **Redis pipelines** — use `transaction=True` for atomicity when doing multi-step Redis operations.
- **`max(0, value)` guards** — for any computed value that should never be negative (remaining credits, limits, etc.).

### SSE / Streaming
- **`data:` lines** for events parsed by the frontend (must match Zod schema).
- **`: comment` lines** for heartbeats, status messages, and non-data events that should be silently discarded by EventSource parsers.

### Testing
- Test files colocated as `*_test.py` next to source files.
- Mock at boundaries — mock where the symbol is **used**, not where it's **defined**.
- After refactoring, update mock targets to match new module paths.
- Use `AsyncMock` for async functions.

## TypeScript / React (Frontend)

### Components
- **Function declarations** for components and handlers — not arrow functions (except small inline lambdas).
- **No unnecessary `useCallback`/`useMemo`** — only add when there's a measured performance issue.
- **No `useEffect` abuse** — derive state directly when possible.

### Styling
- **Tailwind CSS only** — use design tokens, avoid hardcoded values.
- **No `dark:` classes** — the design system handles dark mode.
- **No `<a>` tags** for internal navigation — use Next.js `<Link>`.
- **Phosphor Icons only** — no other icon libraries.

### Data fetching
- **Generated API hooks** (Orval) — never use `BackendAPI` or `src/lib/autogpt-server-api/*`.
- **React Query** for server state, colocated near consumers.

### Code conventions
- Capitalize acronyms in symbols: `graphID`, `useBackendAPI`.
- `interface Props { ... }` (not exported) for component props.
- Avoid barrel files and `index.ts` re-exports.
- No `any` types unless the value genuinely can be anything.

## Review checklist

Before finishing, always ask:
- Can any function be split into smaller pieces?
- Is there unnecessary nesting that an early return would eliminate?
- Can any loop be a comprehension (Python) or `.map`/`.filter` (JS)?
- Is there a simpler way to express this logic?
- Are error messages sanitized (no full file paths)?
- Are mocks targeting the right module paths?
