# Frontend

This file provides guidance to coding agents when working with the frontend.

## Essential Commands

```bash
# Install dependencies
pnpm i

# Generate API client from OpenAPI spec
pnpm generate:api

# Start development server
pnpm dev

# Run E2E tests
pnpm test

# Run Storybook for component development
pnpm storybook

# Build production
pnpm build

# Format and lint
pnpm format

# Type checking
pnpm types
```

### Pre-completion Checks (MANDATORY)

After making **any** code changes in the frontend, you MUST run the following commands **in order** before reporting work as done, creating commits, or opening PRs:

1. `pnpm format` — auto-fix formatting issues
2. `pnpm lint` — check for lint errors; fix any that appear
3. `pnpm types` — check for type errors; fix any that appear

Do NOT skip these steps. If any command reports errors, fix them and re-run until clean. Only then may you consider the task complete. If typing keeps failing, stop and ask the user.

4. `pnpm test:unit` — run integration tests; fix any failures

### Code Style

- Fully capitalize acronyms in symbols, e.g. `graphID`, `useBackendAPI`
- Use function declarations (not arrow functions) for components/handlers
- No `dark:` Tailwind classes — the design system handles dark mode
- Use Next.js `<Link>` for internal navigation — never raw `<a>` tags
- No `any` types unless the value genuinely can be anything
- No linter suppressors (`// @ts-ignore`, `// eslint-disable`) — fix the actual issue
- **File length** — keep files under ~200 lines; extract sub-components or hooks into their own files when a file grows beyond this
- **Function/component length** — keep render functions and hooks under ~50 lines; extract named helpers or sub-components when they grow longer

## Architecture

- **Framework**: Next.js 15 App Router (client-first approach)
- **Data Fetching**: Type-safe generated API hooks via Orval + React Query
- **State Management**: React Query for server state, co-located UI state in components/hooks
- **Component Structure**: Separate render logic (`.tsx`) from business logic (`use*.ts` hooks)
- **Workflow Builder**: Visual graph editor using @xyflow/react
- **UI Components**: shadcn/ui (Radix UI primitives) with Tailwind CSS styling
- **Icons**: Phosphor Icons only
- **Feature Flags**: LaunchDarkly integration
- **Error Handling**: ErrorCard for render errors, toast for mutations, Sentry for exceptions
- **Testing**: Vitest + React Testing Library + MSW for integration tests (primary), Playwright for E2E, Storybook for visual

## Environment Configuration

`.env.default` (defaults) → `.env` (user overrides)

## Feature Development

See @CONTRIBUTING.md for complete patterns. Quick reference:

1. **Pages**: Create in `src/app/(platform)/feature-name/page.tsx`
   - Extract component logic into custom hooks grouped by concern, not by component. Each hook should represent a cohesive domain of functionality (e.g., useSearch, useFilters, usePagination) rather than bundling all state into one useComponentState hook.
     - Put each hook in its own `.ts` file
   - Put sub-components in local `components/` folder
   - Component props should be `type Props = { ... }` (not exported) unless it needs to be used outside the component
2. **Components**: Structure as `ComponentName/ComponentName.tsx` + `useComponentName.ts` + `helpers.ts`
   - Use design system components from `src/components/` (atoms, molecules, organisms)
   - Never use `src/components/__legacy__/*`
3. **Data fetching**: Use generated API hooks from `@/app/api/__generated__/endpoints/`
   - Regenerate with `pnpm generate:api`
   - Pattern: `use{Method}{Version}{OperationName}`
4. **Styling**: Tailwind CSS only, use design tokens, Phosphor Icons only
5. **Testing**: Integration tests are the default (~90%). See `TESTING.md` for full details.
   - **New pages/features**: Write integration tests in `__tests__/` next to `page.tsx` using Vitest + RTL + MSW
   - **API mocking**: Use Orval-generated MSW handlers from `@/app/api/__generated__/endpoints/{tag}/{tag}.msw.ts`
   - **Run**: `pnpm test:unit` (integration/unit), `pnpm test` (Playwright E2E)
   - **Storybook**: For design system components in `src/components/`
   - **TDD**: Write a failing test first, implement, then verify
6. **Code conventions**:
   - Use function declarations (not arrow functions) for components/handlers
   - Do not use `useCallback` or `useMemo` unless asked to optimise a given function
   - Do not type hook returns, let Typescript infer as much as possible
   - Never type with `any` unless a variable/attribute can ACTUALLY be of any type
   - avoid index and barrel files
