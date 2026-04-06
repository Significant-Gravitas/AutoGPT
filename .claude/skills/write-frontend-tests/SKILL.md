---
name: write-frontend-tests
description: "Analyze the current branch diff against dev, plan integration tests for changed frontend pages/components, and write them. TRIGGER when user asks to write frontend tests, add test coverage, or 'write tests for my changes'."
user-invocable: true
args: "[base branch] — defaults to dev. Optionally pass a specific base branch to diff against."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Write Frontend Tests

Analyze the current branch's frontend changes, plan integration tests, and write them.

## References

Before writing any tests, read the testing rules and conventions:

- `autogpt_platform/frontend/TESTING.md` — testing strategy, file locations, examples
- `autogpt_platform/frontend/src/tests/AGENTS.md` — detailed testing rules, MSW patterns, decision flowchart
- `autogpt_platform/frontend/src/tests/integrations/test-utils.tsx` — custom render with providers
- `autogpt_platform/frontend/src/tests/integrations/vitest.setup.tsx` — MSW server setup

## Step 1: Identify changed frontend files

```bash
BASE_BRANCH="${ARGUMENTS:-dev}"
cd autogpt_platform/frontend

# Get changed frontend files (excluding generated, config, and test files)
git diff "$BASE_BRANCH"...HEAD --name-only -- src/ \
  | grep -v '__generated__' \
  | grep -v '__tests__' \
  | grep -v '\.test\.' \
  | grep -v '\.stories\.' \
  | grep -v '\.spec\.'
```

Also read the diff to understand what changed:

```bash
git diff "$BASE_BRANCH"...HEAD --stat -- src/
git diff "$BASE_BRANCH"...HEAD -- src/ | head -500
```

## Step 2: Categorize changes and find test targets

For each changed file, determine:

1. **Is it a page?** (`page.tsx`) — these are the primary test targets
2. **Is it a hook?** (`use*.ts`) — test via the page that uses it
3. **Is it a component?** (`.tsx` in `components/`) — test via the parent page unless it's complex enough to warrant isolation
4. **Is it a helper?** (`helpers.ts`, `utils.ts`) — unit test directly if pure logic

**Priority order:**
1. Pages with new/changed data fetching or user interactions
2. Components with complex internal logic (modals, forms, wizards)
3. Hooks with non-trivial business logic
4. Pure helper functions

Skip: styling-only changes, type-only changes, config changes.

## Step 3: Check for existing tests

For each test target, check if tests already exist:

```bash
# For a page at src/app/(platform)/library/page.tsx
ls src/app/\(platform\)/library/__tests__/ 2>/dev/null

# For a component at src/app/(platform)/library/components/AgentCard/AgentCard.tsx
ls src/app/\(platform\)/library/components/AgentCard/__tests__/ 2>/dev/null
```

Note which targets have no tests (need new files) vs which have tests that need updating.

## Step 4: Identify API endpoints used

For each test target, find which API hooks are used:

```bash
# Find generated API hook imports in the changed files
grep -rn 'from.*__generated__/endpoints' src/app/\(platform\)/library/
grep -rn 'use[A-Z].*V[12]' src/app/\(platform\)/library/
```

For each API hook found, locate the corresponding MSW handler:

```bash
# If the page uses useGetV2ListLibraryAgents, find its MSW handlers
grep -rn 'getGetV2ListLibraryAgents.*Handler' src/app/api/__generated__/endpoints/library/library.msw.ts
```

List every MSW handler you will need (200 for happy path, 4xx for error paths).

## Step 5: Write the test plan

Before writing code, output a plan as a numbered list:

```
Test plan for [branch name]:

1. src/app/(platform)/library/__tests__/main.test.tsx (NEW)
   - Renders page with agent list (MSW 200)
   - Shows loading state
   - Shows error state (MSW 422)
   - Handles empty agent list

2. src/app/(platform)/library/__tests__/search.test.tsx (NEW)
   - Filters agents by search query
   - Shows no results message
   - Clears search

3. src/app/(platform)/library/components/AgentCard/__tests__/AgentCard.test.tsx (UPDATE)
   - Add test for new "duplicate" action
```

Present this plan to the user. Wait for confirmation before proceeding. If the user has feedback, adjust the plan.

## Step 6: Write the tests

For each test file in the plan, follow these conventions:

### File structure

```tsx
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
// Import MSW handlers for endpoints the page uses
import {
  getGetV2ListLibraryAgentsMockHandler200,
  getGetV2ListLibraryAgentsMockHandler422,
} from "@/app/api/__generated__/endpoints/library/library.msw";
// Import the component under test
import LibraryPage from "../page";

describe("LibraryPage", () => {
  test("renders agent list from API", async () => {
    server.use(getGetV2ListLibraryAgentsMockHandler200());

    render(<LibraryPage />);

    expect(await screen.findByText(/my agents/i)).toBeDefined();
  });

  test("shows error state on API failure", async () => {
    server.use(getGetV2ListLibraryAgentsMockHandler422());

    render(<LibraryPage />);

    expect(await screen.findByText(/error/i)).toBeDefined();
  });
});
```

### Rules

- Use `render()` from `@/tests/integrations/test-utils` (NOT from `@testing-library/react` directly)
- Use `server.use()` to set up MSW handlers BEFORE rendering
- Use `findBy*` (async) for elements that appear after data fetching — NOT `getBy*`
- Use `getBy*` only for elements that are immediately present in the DOM
- Use `screen` queries — do NOT destructure from `render()`
- Use `waitFor` when asserting side effects or state changes after interactions
- Import `fireEvent` or `userEvent` from the test-utils for interactions
- Do NOT mock internal hooks or functions — mock at the API boundary via MSW
- Do NOT use `act()` manually — `render` and `fireEvent` handle it
- Keep tests focused: one behavior per test
- Use descriptive test names that read like sentences

### Test location

```
# For pages: __tests__/ next to page.tsx
src/app/(platform)/library/__tests__/main.test.tsx

# For complex standalone components: __tests__/ inside component folder
src/app/(platform)/library/components/AgentCard/__tests__/AgentCard.test.tsx

# For pure helpers: co-located .test.ts
src/app/(platform)/library/helpers.test.ts
```

### Custom MSW overrides

When the auto-generated faker data is not enough, override with specific data:

```tsx
import { http, HttpResponse } from "msw";

server.use(
  http.get("http://localhost:3000/api/proxy/api/v2/library/agents", () => {
    return HttpResponse.json({
      agents: [
        { id: "1", name: "Test Agent", description: "A test agent" },
      ],
      pagination: { total_items: 1, total_pages: 1, page: 1, page_size: 10 },
    });
  }),
);
```

Use the proxy URL pattern: `http://localhost:3000/api/proxy/api/v{version}/{path}` — this matches the MSW base URL configured in `orval.config.ts`.

## Step 7: Run and verify

After writing all tests:

```bash
cd autogpt_platform/frontend
pnpm test:unit --reporter=verbose
```

If tests fail:
1. Read the error output carefully
2. Fix the test (not the source code, unless there is a genuine bug)
3. Re-run until all pass

Then run the full checks:

```bash
pnpm format
pnpm lint
pnpm types
```
