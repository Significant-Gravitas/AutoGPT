# Frontend Testing Rules 🧪

## Testing Types Overview

| Type            | Tool                  | Speed           | Purpose                          |
| --------------- | --------------------- | --------------- | -------------------------------- |
| **E2E**         | Playwright            | Slow (~5s/test) | Real browser, full user journeys |
| **Integration** | Vitest + RTL          | Fast (~100ms)   | Component + mocked API           |
| **Unit**        | Vitest + RTL          | Fastest (~10ms) | Individual functions/components  |
| **Visual**      | Storybook + Chromatic | N/A             | UI appearance, design system     |

---

## When to Use Each

### ✅ E2E Tests (Playwright)

**Use for:** Critical user journeys that MUST work in a real browser.

- Authentication flows (login, signup, logout)
- Payment or sensitive transactions
- Flows requiring real browser APIs (clipboard, downloads)
- Cross-page navigation that must work end-to-end

**Location:** `src/tests/*.spec.ts` (centralized, as there will be fewer of them)

### ✅ Integration Tests (Vitest + RTL)

**Use for:** Testing components with their dependencies (API calls, state).

- Page-level behavior with mocked API responses
- Components that fetch data
- User interactions that trigger API calls
- Feature flows within a single page

**Location:** Place tests in a `__tests__` folder next to the component:

```
ComponentName/
  __tests__/
    main.test.tsx
    some-flow.test.tsx
  ComponentName.tsx
  useComponentName.ts
```

**Start at page level:** Initially write integration tests at the "page" level. No need to write them for every small component.

```
/library/
  __tests__/
    main.test.tsx
    searching-agents.test.tsx
    agents-pagination.test.tsx
  page.tsx
  useLibraryPage.ts
```

Start with a `main.test.tsx` file and split into smaller files as it grows.

**What integration tests should do:**

1. Render a page or complex modal (e.g., `AgentPublishModal`)
2. Mock API requests via MSW
3. Assert UI scenarios via Testing Library

```tsx
// Example: Test page renders data from API
import { server } from "@/mocks/mock-server";
import { getDeleteV2DeleteStoreSubmissionMockHandler422 } from "@/app/api/__generated__/endpoints/store/store.msw";

test("shows error when submission fails", async () => {
  // Override default handler to return error status
  server.use(getDeleteV2DeleteStoreSubmissionMockHandler422());

  render(<MarketplacePage />);
  await screen.findByText("Featured Agents");
  // ... assert error UI
});
```

**Tip:** Use `findBy...` methods most of the time—they wait for elements to appear, so async code won't cause flaky tests. The regular `getBy...` methods don't wait and error immediately.

### ✅ Unit Tests (Vitest + RTL)

**Use for:** Testing isolated components and utility functions.

- Pure utility functions (`lib/utils.ts`)
- Component rendering with various props
- Component state changes
- Custom hooks

**Location:** Co-located with the file: `Component.test.tsx` next to `Component.tsx`

```tsx
// Example: Test component renders correctly
render(<AgentCard title="My Agent" />);
expect(screen.getByText("My Agent")).toBeInTheDocument();
```

### ✅ Storybook Tests (Visual)

**Use for:** Design system, visual appearance, component documentation.

- Atoms (Button, Input, Badge)
- Molecules (Dialog, Card)
- Visual states (hover, disabled, loading)
- Responsive layouts

**Location:** Co-located: `Component.stories.tsx` next to `Component.tsx`

---

## Decision Flowchart

```
Does it need a REAL browser/backend?
├─ YES → E2E (Playwright)
└─ NO
   └─ Does it involve API calls or complex state?
      ├─ YES → Integration (Vitest + RTL)
      └─ NO
         └─ Is it about visual appearance?
            ├─ YES → Storybook
            └─ NO → Unit (Vitest + RTL)
```

---

## What NOT to Test

❌ Third-party library internals (Radix UI, React Query)  
❌ CSS styling details (use Storybook)  
❌ Simple prop-passing components with no logic  
❌ TypeScript types

---

## File Organization

```
src/
├── components/
│   └── atoms/
│       └── Button/
│           ├── Button.tsx
│           ├── Button.test.tsx      # Unit test
│           └── Button.stories.tsx   # Visual test
├── app/
│   └── (platform)/
│       └── marketplace/
│           └── components/
│               └── MainMarketplacePage/
│                   ├── __tests__/
│                   │   ├── main.test.tsx           # Integration test
│                   │   └── search-agents.test.tsx  # Integration test
│                   ├── MainMarketplacePage.tsx
│                   └── useMainMarketplacePage.ts
├── lib/
│   ├── utils.ts
│   └── utils.test.ts                # Unit test
├── mocks/
│   ├── mock-handlers.ts             # MSW handlers (auto-generated via Orval)
│   └── mock-server.ts               # MSW server setup
└── tests/
    ├── integrations/
    │   ├── test-utils.tsx           # Testing utilities
    │   └── vitest.setup.tsx         # Integration test setup
    └── *.spec.ts                    # E2E tests (Playwright) - centralized
```

---

## Priority Matrix

| Component Type      | Test Priority | Recommended Test |
| ------------------- | ------------- | ---------------- |
| Pages/Features      | **Highest**   | Integration      |
| Custom Hooks        | High          | Unit             |
| Utility Functions   | High          | Unit             |
| Organisms (complex) | High          | Integration      |
| Molecules           | Medium        | Unit + Storybook |
| Atoms               | Medium        | Storybook only\* |

\*Atoms are typically simple enough that Storybook visual tests suffice.

---

## MSW Mocking

API mocking is handled via MSW (Mock Service Worker). Handlers are auto-generated by Orval from the OpenAPI schema.

**Default behavior:** All client-side requests are intercepted and return 200 status with faker-generated data.

**Override for specific tests:** Use generated error handlers to test non-OK status scenarios:

```tsx
import { server } from "@/mocks/mock-server";
import { getDeleteV2DeleteStoreSubmissionMockHandler422 } from "@/app/api/__generated__/endpoints/store/store.msw";

test("shows error when deletion fails", async () => {
  server.use(getDeleteV2DeleteStoreSubmissionMockHandler422());

  render(<MyComponent />);
  // ... assert error UI
});
```

**Generated handlers location:** `src/app/api/__generated__/endpoints/*/` - each endpoint has handlers for different status codes.

---

## Golden Rules

1. **Test behavior, not implementation** - Query by role/text, not class names
2. **One assertion per concept** - Tests should be focused
3. **Mock at boundaries** - Mock API calls, not internal functions
4. **Co-locate integration tests** - Keep `__tests__/` folder next to the component
5. **E2E is expensive** - Only for critical happy paths; prefer integration tests
6. **AI agents are good at writing integration tests** - Start with these when adding test coverage
