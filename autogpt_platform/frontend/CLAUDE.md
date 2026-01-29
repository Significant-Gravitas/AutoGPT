# CLAUDE.md - Frontend

This file provides guidance to Claude Code when working with the frontend.

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

### Code Style

- Fully capitalize acronyms in symbols, e.g. `graphID`, `useBackendAPI`
- Use function declarations (not arrow functions) for components/handlers

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
- **Testing**: Playwright for E2E, Storybook for component development

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
5. **Testing**: Add Storybook stories for new components, Playwright for E2E
6. **Code conventions**:
   - Use function declarations (not arrow functions) for components/handlers
   - Do not use `useCallback` or `useMemo` unless asked to optimise a given function
   - Do not type hook returns, let Typescript infer as much as possible
   - Never type with `any` unless a variable/attribute can ACTUALLY be of any type
