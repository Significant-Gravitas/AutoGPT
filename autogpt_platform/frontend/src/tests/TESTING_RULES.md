# Frontend Testing Rules 🧪

## Testing Types Overview

| Type | Tool | Speed | Purpose |
|------|------|-------|---------|
| **E2E** | Playwright | Slow (~5s/test) | Real browser, full user journeys |
| **Integration** | Vitest + RTL | Fast (~100ms) | Component + mocked API |
| **Unit** | Vitest + RTL | Fastest (~10ms) | Individual functions/components |
| **Visual** | Storybook + Chromatic | N/A | UI appearance, design system |

---

## When to Use Each

### ✅ E2E Tests (Playwright)
**Use for:** Critical user journeys that MUST work in a real browser.

- Authentication flows (login, signup, logout)
- Payment or sensitive transactions
- Flows requiring real browser APIs (clipboard, downloads)
- Cross-page navigation that must work end-to-end

**Location:** `src/tests/*.spec.ts`

### ✅ Integration Tests (Vitest + RTL)
**Use for:** Testing components with their dependencies (API calls, state).

- Page-level behavior with mocked API responses
- Components that fetch data
- User interactions that trigger API calls
- Feature flows within a single page

**Location:** `src/app/**/page.test.tsx` or `src/components/**/Component.test.tsx`

```tsx
// Example: Test page renders data from API
render(<MarketplacePage />, { wrapper: MockProviders });
await screen.findByText('Featured Agents');
expect(screen.getByRole('list')).toHaveLength(3);
```

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
expect(screen.getByText('My Agent')).toBeInTheDocument();
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
│           ├── page.tsx
│           └── page.test.tsx        # Integration test
├── lib/
│   ├── utils.ts
│   └── utils.test.ts                # Unit test
└── tests/
    └── *.spec.ts                    # E2E tests (Playwright)
```

---

## Priority Matrix

| Component Type | Test Priority | Recommended Test |
|----------------|---------------|------------------|
| Pages/Features | **Highest** | Integration |
| Custom Hooks | High | Unit |
| Utility Functions | High | Unit |
| Organisms (complex) | High | Integration |
| Molecules | Medium | Unit + Storybook |
| Atoms | Medium | Storybook only* |

*Atoms are typically simple enough that Storybook visual tests suffice.

---

## Golden Rules

1. **Test behavior, not implementation** - Query by role/text, not class names
2. **One assertion per concept** - Tests should be focused
3. **Mock at boundaries** - Mock API calls, not internal functions
4. **Co-locate tests** - Keep `.test.tsx` next to the component
5. **E2E is expensive** - Only for critical journeys; prefer integration tests
