<div align="center">
  <h1>AutoGPT Frontend • Contributing ⌨️</h1>
  <p>Next.js App Router • Client-first • Type-safe generated API hooks • Tailwind + shadcn/ui</p>
</div>

---

## ☕️ Summary

This document is your reference for contributing to the AutoGPT Frontend. It adapts legacy guidelines to our current stack and practices.

- Architecture and stack
- Component structure and design system
- Data fetching (generated API hooks)
- Feature flags
- Naming and code conventions
- Tooling, scripts, and testing
- PR process and checklist

This is a living document. Open a pull request any time to improve it.

---

## 🚀 Quick Start FAQ

New to the codebase? Here are shortcuts to common tasks:

### I need to make a new page

1. Create page in `src/app/(platform)/your-feature/page.tsx`
2. If it has logic, create `usePage.ts` hook next to it
3. Create sub-components in `components/` folder
4. Use generated API hooks for data fetching
5. If page needs auth, ensure it's in the `(platform)` route group

**Example structure:**

```
app/(platform)/dashboard/
  page.tsx
  useDashboardPage.ts
  components/
    StatsPanel/
      StatsPanel.tsx
      useStatsPanel.ts
```

See [Component structure](#-component-structure) and [Styling](#-styling) and [Data fetching patterns](#-data-fetching-patterns) sections.

### I need to update an existing component in a page

1. Find the page `src/app/(platform)/your-feature/page.tsx`
2. Check its `components/` folder
3. If needing to update its logic, check the `use[Component].ts` hook
4. If the update is related to rendering, check `[Component].tsx` file

See [Component structure](#-component-structure) and [Styling](#-styling) sections.

### I need to make a new API call and show it on the UI

1. Ensure the backend endpoint exists in the OpenAPI spec
2. Regenerate API client: `pnpm generate:api`
3. Import the generated hook by typing the operation name (auto-import)
4. Use the hook in your component/custom hook
5. Handle loading, error, and success states

**Example:**

```tsx
import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";

export function useAgentList() {
  const { data, isLoading, isError, error } = useGetV2ListLibraryAgents();

  return {
    agents: data?.data || [],
    isLoading,
    isError,
    error,
  };
}
```

See [Data fetching patterns](#-data-fetching-patterns) for more examples.

### I need to create a new component in the Design System

1. Determine the atomic level: atom, molecule, or organism
2. Create folder: `src/components/[level]/ComponentName/`
3. Create `ComponentName.tsx` (render logic)
4. If logic exists, create `useComponentName.ts`
5. Create `ComponentName.stories.tsx` for Storybook
6. Use Tailwind + design tokens (avoid hardcoded values)
7. Only use Phosphor icons
8. Test in Storybook: `pnpm storybook`
9. Verify in Chromatic after PR

**Example structure:**

```
src/components/molecules/DataCard/
  DataCard.tsx
  DataCard.stories.tsx
  useDataCard.ts
```

See [Component structure](#-component-structure) and [Styling](#-styling) sections.

---

## 📟 Contribution process

### 1) Branch off `dev`

- Branch from `dev` for features and fixes
- Keep PRs focused (aim for one ticket per PR)
- Use conventional commit messages with a scope (e.g., `feat(frontend): add X`)

### 2) Feature flags

If a feature will ship across multiple PRs, guard it with a flag so we can merge iteratively.

- Use [LaunchDarkly](https://www.launchdarkly.com) based flags (see Feature Flags below)
- Avoid long-lived feature branches

### 3) Open PR and get reviews ✅

Before requesting review:

- [x] Code follows architecture and conventions here
- [x] `pnpm format && pnpm lint && pnpm types` pass
- [x] Relevant tests pass locally: `pnpm test` (and/or Storybook tests)
- [x] If touching UI, validate against our design system and stories

### 4) Merge to `dev`

- Use squash merges
- Follow conventional commit message format for the squash title

---

## 📂 Architecture & Stack

### Next.js App Router

- We use the [Next.js App Router](https://nextjs.org/docs/app) in `src/app`
- Use [route segments](https://nextjs.org/docs/app/building-your-application/routing) with semantic URLs; no `pages/`

### Component good practices

- Default to client components
- Use server components only when:
  - SEO requires server-rendered HTML, or
  - Extreme first-byte performance justifies it
  - If you render server-side data, prefer server-side prefetch + client hydration (see examples below and [React Query SSR & Hydration](https://tanstack.com/query/latest/docs/framework/react/guides/ssr))
- Prefer using [Next.js API routes](https://nextjs.org/docs/pages/building-your-application/routing/api-routes) when possible over [server actions](https://nextjs.org/docs/14/app/building-your-application/data-fetching/server-actions-and-mutations)
- Keep components small and simple
  - favour composition and splitting large components into smaller bits of UI
  - [colocate state](https://kentcdodds.com/blog/state-colocation-will-make-your-react-app-faster) when possible
  - keep render/side-effects split for [separation of concerns](https://en.wikipedia.org/wiki/Separation_of_concerns)
  - do not over-complicate or re-invent the wheel

**❓ Why a client-side first design vs server components/actions?**

While server components and actions are cool and cutting-edge, they introduce a layer of complexity which not always justified by the benefits they deliver. Defaulting to client-first keeps things simple in the mental model of the developer, specially for those developers less familiar with Next.js or heavy Front-end development.

### Data fetching: prefer generated API hooks

- We generate a type-safe client and React Query hooks from the backend OpenAPI spec via [Orval](https://orval.dev/)
- Prefer the generated hooks under `src/app/api/__generated__/endpoints/...`
- Treat `BackendAPI` and code under `src/lib/autogpt-server-api/*` as deprecated; do not introduce new usages
- Use [Zod](https://zod.dev/) schemas from the generated client where applicable

### State management

- Prefer [React Query](https://tanstack.com/query/latest/docs/framework/react/overview) for server state, colocated near consumers (see [state colocation](https://kentcdodds.com/blog/state-colocation-will-make-your-react-app-faster))
- Co-locate UI state inside components/hooks; keep global state minimal
- Avoid `useMemo` and `useCallback` unless you have a measured performance issue
- Do not abuse `useEffect`; prefer state colocation and derive values directly when possible

### Styling and components

- [Tailwind CSS](https://tailwindcss.com/docs) + [shadcn/ui](https://ui.shadcn.com/) ([Radix Primitives](https://www.radix-ui.com/docs/primitives/overview/introduction) under the hood)
- Use the design system under `src/components` for primitives and building blocks
- Do not use anything under `src/components/_legacy__`; migrate away from it when touching old code
- Reference the design system catalog on Chromatic: [`https://dev--670f94474adee5e32c896b98.chromatic.com/`](https://dev--670f94474adee5e32c896b98.chromatic.com/)
- Use the [`tailwind-scrollbar`](https://www.npmjs.com/package/tailwind-scrollbar) plugin utilities for scrollbar styling

---

## 🧱 Component structure

For components, separate render logic from data/behavior, and keep implementation details local.

**Most components should follow this structure.** Pages are just bigger components made of smaller ones, and sub-components can have their own nested sub-components when dealing with complex features.

### Basic structure

When a component has non-trivial logic:

```
FeatureX/
  FeatureX.tsx        (render logic only)
  useFeatureX.ts      (hook; data fetching, behavior, state)
  helpers.ts          (pure helpers used by the hook)
  components/         (optional, subcomponents local to FeatureX)
```

### Example: Page with nested components

```tsx
// Page composition
app/(platform)/dashboard/
  page.tsx
  useDashboardPage.ts
    components/ # (Sub-components the dashboard page is made of)
      StatsPanel/
        StatsPanel.tsx
        useStatsPanel.ts
        helpers.ts
        components/ # (Sub-components belonging to StatsPanel)
          StatCard/
            StatCard.tsx
      ActivityFeed/
        ActivityFeed.tsx
        useActivityFeed.ts
```

### Guidelines

- Prefer function declarations for components and handlers
- Only use arrow functions for small inline lambdas (e.g., in `map`)
- Avoid barrel files and `index.ts` re-exports
- Keep component files focused and readable; push complex logic to `helpers.ts`
- Abstract reusable, cross-feature logic into `src/services/` or `src/lib/utils.ts` as appropriate
- Build components encapsulated so they can be easily reused and abstracted elsewhere
- Nest sub-components within a `components/` folder when they're local to the parent feature

### Exceptions

When to simplify the structure:

**Small hook logic (3-4 lines)**

If the hook logic is minimal, keep it inline with the render function:

```tsx
export function ActivityAlert() {
  const [isVisible, setIsVisible] = useState(true);
  if (!isVisible) return null;

  return (
    <Alert onClose={() => setIsVisible(false)}>New activity detected</Alert>
  );
}
```

**Render-only components**

Components with no hook logic can be direct files in `components/` without a folder:

```
components/
  ActivityAlert.tsx      (render-only, no folder needed)
  StatsPanel/            (has hook logic, needs folder)
    StatsPanel.tsx
    useStatsPanel.ts
```

### Hook file structure

When separating logic into a custom hook:

```tsx
// useStatsPanel.ts
export function useStatsPanel() {
  const [data, setData] = useState<Stats[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchStats().then(setData);
  }, []);

  return {
    data,
    isLoading,
    refresh: () => fetchStats().then(setData),
  };
}
```

Rules:

- **Always return an object** that exposes data and methods to the view
- **Export a single function** named after the component (e.g., `useStatsPanel` for `StatsPanel.tsx`)
- **Abstract into helpers.ts** when hook logic grows large, so the hook file remains readable by scanning without diving into implementation details

---

## 🔄 Data fetching patterns

All API hooks are generated from the backend OpenAPI specification using [Orval](https://orval.dev/). The hooks are type-safe and follow the operation names defined in the backend API.

### How to discover hooks

Most of the time you can rely on auto-import by typing the endpoint or operation name. Your IDE will suggest the generated hooks based on the OpenAPI operation IDs.

**Examples of hook naming patterns:**

- `GET /api/v1/notifications` → `useGetV1GetNotificationPreferences`
- `POST /api/v2/store/agents` → `usePostV2CreateStoreAgent`
- `DELETE /api/v2/store/submissions/{id}` → `useDeleteV2DeleteStoreSubmission`
- `GET /api/v2/library/agents` → `useGetV2ListLibraryAgents`

**Pattern**: `use{Method}{Version}{OperationName}`

You can also explore the generated hooks by browsing `src/app/api/__generated__/endpoints/` which is organized by API tags (e.g., `auth`, `store`, `library`).

**OpenAPI specs:**

- Production: [https://backend.agpt.co/openapi.json](https://backend.agpt.co/openapi.json)
- Staging: [https://dev-server.agpt.co/openapi.json](https://dev-server.agpt.co/openapi.json)

### Generated hooks (client)

Prefer the generated React Query hooks (via Orval + React Query):

```tsx
import { useGetV1GetNotificationPreferences } from "@/app/api/__generated__/endpoints/auth/auth";

export function PreferencesPanel() {
  const { data, isLoading, isError } = useGetV1GetNotificationPreferences({
    query: {
      select: (res) => res.data,
    },
  });

  if (isLoading) return null;
  if (isError) throw new Error("Failed to load preferences");
  return <pre>{JSON.stringify(data, null, 2)}</pre>;
}
```

### Generated mutations (client)

```tsx
import { useQueryClient } from "@tanstack/react-query";
import {
  useDeleteV2DeleteStoreSubmission,
  getGetV2ListMySubmissionsQueryKey,
} from "@/app/api/__generated__/endpoints/store/store";

export function DeleteSubmissionButton({
  submissionId,
}: {
  submissionId: string;
}) {
  const queryClient = useQueryClient();
  const { mutateAsync: deleteSubmission, isPending } =
    useDeleteV2DeleteStoreSubmission({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListMySubmissionsQueryKey(),
          });
        },
      },
    });

  async function onClick() {
    await deleteSubmission({ submissionId });
  }

  return (
    <button disabled={isPending} onClick={onClick}>
      Delete
    </button>
  );
}
```

### Server-side prefetch + client hydration

Use server-side prefetch to improve TTFB while keeping the component tree client-first (see [React Query SSR & Hydration](https://tanstack.com/query/latest/docs/framework/react/guides/ssr)):

```tsx
// in a server component
import { getQueryClient } from "@/lib/tanstack-query/getQueryClient";
import { HydrationBoundary, dehydrate } from "@tanstack/react-query";
import {
  prefetchGetV2ListStoreAgentsQuery,
  prefetchGetV2ListStoreCreatorsQuery,
} from "@/app/api/__generated__/endpoints/store/store";

export default async function MarketplacePage() {
  const queryClient = getQueryClient();

  await Promise.all([
    prefetchGetV2ListStoreAgentsQuery(queryClient, { featured: true }),
    prefetchGetV2ListStoreAgentsQuery(queryClient, { sorted_by: "runs" }),
    prefetchGetV2ListStoreCreatorsQuery(queryClient, {
      featured: true,
      sorted_by: "num_agents",
    }),
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      {/* Client component tree goes here */}
    </HydrationBoundary>
  );
}
```

Notes:

- Do not introduce new usages of `BackendAPI` or `src/lib/autogpt-server-api/*`
- Keep transformations and mapping logic close to the consumer (hook), not in the view

---

## ⚠️ Error handling

The app has multiple error handling strategies depending on the type of error:

### Render/runtime errors

Use `<ErrorCard />` to display render or runtime errors gracefully:

```tsx
import { ErrorCard } from "@/components/molecules/ErrorCard";

export function DataPanel() {
  const { data, isLoading, isError, error } = useGetData();

  if (isLoading) return <Skeleton />;
  if (isError) return <ErrorCard error={error} />;

  return <div>{data.content}</div>;
}
```

### API mutation errors

Display mutation errors using toast notifications:

```tsx
import { useToast } from "@/components/ui/use-toast";

export function useUpdateSettings() {
  const { toast } = useToast();
  const { mutateAsync: updateSettings } = useUpdateSettingsMutation({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to update settings",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  return { updateSettings };
}
```

### Manual Sentry capture

When needed, you can manually capture exceptions to Sentry:

```tsx
import * as Sentry from "@sentry/nextjs";

try {
  await riskyOperation();
} catch (error) {
  Sentry.captureException(error, {
    tags: { context: "feature-x" },
    extra: { metadata: additionalData },
  });
  throw error;
}
```

### Global error boundaries

The app has error boundaries already configured to:

- Capture uncaught errors globally and send them to Sentry
- Display a user-friendly error UI when something breaks
- Prevent the entire app from crashing

You don't need to wrap components in error boundaries manually unless you need custom error recovery logic.

---

## 🚩 Feature Flags

- Flags are powered by [LaunchDarkly](https://docs.launchdarkly.com/)
- Use the helper APIs under `src/services/feature-flags`

Check a flag in a client component:

```tsx
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export function AgentActivityPanel() {
  const enabled = useGetFlag(Flag.AGENT_ACTIVITY);
  if (!enabled) return null;
  return <div>Feature is enabled!</div>;
}
```

Protect a route or page component:

```tsx
import { withFeatureFlag } from "@/services/feature-flags/with-feature-flag";

export const MyFeaturePage = withFeatureFlag(function Page() {
  return <div>My feature page</div>;
}, "my-feature-flag");
```

Local dev and Playwright:

- Set `NEXT_PUBLIC_PW_TEST=true` to use mocked flag values during local development and tests

Adding new flags:

1. Add the flag to the `Flag` enum and `FlagValues` type
2. Provide a mock value in the mock map
3. Configure the flag in LaunchDarkly

---

## 📙 Naming conventions

General:

- Variables and functions should read like plain English
- Prefer `const` over `let` unless reassignment is required
- Use searchable constants instead of magic numbers

Files:

- Components and hooks: `PascalCase` for component files, `camelCase` for hooks
- Other files: `kebab-case`
- Do not create barrel files or `index.ts` re-exports

Types:

- Prefer `interface` for object shapes
- Component props should be `interface Props { ... }` (not exported)
- Only use specific exported names (e.g., `export interface MyComponentProps`) when the interface needs to be used outside the component
- Keep type definitions inline with the component - do not create separate `types.ts` files unless types are shared across multiple files
- Use precise types; avoid `any` and unsafe casts

**Props naming examples:**

```tsx
// ✅ Good - internal props, not exported
interface Props {
  title: string;
  onClose: () => void;
}

export function Modal({ title, onClose }: Props) {
  // ...
}

// ✅ Good - exported when needed externally
export interface ModalProps {
  title: string;
  onClose: () => void;
}

export function Modal({ title, onClose }: ModalProps) {
  // ...
}

// ❌ Bad - unnecessarily specific name for internal use
interface ModalComponentProps {
  title: string;
  onClose: () => void;
}

// ❌ Bad - separate types.ts file for single component
// types.ts
export interface ModalProps { ... }

// Modal.tsx
import type { ModalProps } from './types';
```

Parameters:

- If more than one parameter is needed, pass a single `Args` object for clarity

Comments:

- Keep comments minimal; code should be clear by itself
- Only document non-obvious intent, invariants, or caveats

Functions:

- Prefer function declarations for components and handlers
- Only use arrow functions for small inline callbacks

Control flow:

- Use early returns to reduce nesting
- Avoid catching errors unless you handle them meaningfully

---

## 🎨 Styling

- Use Tailwind utilities; prefer semantic, composable class names
- Use shadcn/ui components as building blocks when available
- Use the `tailwind-scrollbar` utilities for scrollbar styling
- Keep responsive and dark-mode behavior consistent with the design system

Additional requirements:

- Do not import shadcn primitives directly in feature code; only use components exposed in our design system under `src/components`. shadcn is a low-level skeleton we style on top of and is not meant to be consumed directly.
- Prefer design tokens over Tailwind's default theme whenever possible (e.g., color, spacing, radius, and typography tokens). Avoid hardcoded values and default palette if a token exists.

---

## ⚠️ Errors and ⏳ Loading

- **Errors**: Use the `ErrorCard` component from the design system to display API/HTTP errors and retry actions. Keep error derivation/mapping in hooks; pass the final message to the component.
  - Component: `src/components/molecules/ErrorCard/ErrorCard.tsx`
- **Loading**: Use the `Skeleton` component(s) from the design system for loading states. Favor domain-appropriate skeleton layouts (lists, cards, tables) over spinners.
  - See Storybook examples under Atoms/Skeleton for patterns.

---

## 🧭 Responsive and mobile-first

- Build mobile-first. Ensure new UI looks great from a 375px viewport width (iPhone SE) upwards.
- Validate layouts at common breakpoints (375, 768, 1024, 1280). Prefer stacking and progressive disclosure on small screens.

---

## 🧰 State for complex flows

For components/flows with complex state, multi-step wizards, or cross-component coordination, prefer a small co-located store using [Zustand](https://github.com/pmndrs/zustand).

Guidelines:

- Co-locate the store with the feature (e.g., `FeatureX/store.ts`).
- Expose typed selectors to minimize re-renders.
- Keep effects and API calls in hooks; stores hold state and pure actions.

Example: simple store with selectors

```ts
import { create } from "zustand";

interface WizardState {
  step: number;
  data: Record<string, unknown>;
  next(): void;
  back(): void;
  setField(args: { key: string; value: unknown }): void;
}

export const useWizardStore = create<WizardState>((set) => ({
  step: 0,
  data: {},
  next() {
    set((state) => ({ step: state.step + 1 }));
  },
  back() {
    set((state) => ({ step: Math.max(0, state.step - 1) }));
  },
  setField({ key, value }) {
    set((state) => ({ data: { ...state.data, [key]: value } }));
  },
}));

// Usage in a component (selectors keep updates scoped)
function WizardFooter() {
  const step = useWizardStore((s) => s.step);
  const next = useWizardStore((s) => s.next);
  const back = useWizardStore((s) => s.back);

  return (
    <div className="flex items-center gap-2">
      <button onClick={back} disabled={step === 0}>Back</button>
      <button onClick={next}>Next</button>
    </div>
  );
}
```

Example: async action coordinated via hook + store

```ts
// FeatureX/useFeatureX.ts
import { useMutation } from "@tanstack/react-query";
import { useWizardStore } from "./store";

export function useFeatureX() {
  const setField = useWizardStore((s) => s.setField);
  const next = useWizardStore((s) => s.next);

  const { mutateAsync: save, isPending } = useMutation({
    mutationFn: async (payload: unknown) => {
      // call API here
      return payload;
    },
    onSuccess(data) {
      setField({ key: "result", value: data });
      next();
    },
  });

  return { save, isSaving: isPending };
}
```

---

## 🖼 Icons

- Only use Phosphor Icons. Treat all other icon libraries as deprecated for new code.
  - Package: `@phosphor-icons/react`
  - Site: [`https://phosphoricons.com/`](https://phosphoricons.com/)

Example usage:

```tsx
import { Plus } from "@phosphor-icons/react";

export function CreateButton() {
  return (
    <button type="button" className="inline-flex items-center gap-2">
      <Plus size={16} />
      Create
    </button>
  );
}
```

---

## 🧪 Testing & Storybook

- See `TESTING.md` for Playwright setup, E2E data seeding, and Storybook usage.

---

## 🛠 Tooling & Scripts

Common scripts (see `package.json` for full list):

- `pnpm dev` — Start Next.js dev server (generates API client first)
- `pnpm build` — Build for production
- `pnpm start` — Start production server
- `pnpm lint` — ESLint + Prettier check
- `pnpm format` — Format code
- `pnpm types` — Type-check
- `pnpm storybook` — Run Storybook
- `pnpm test` — Run Playwright tests

Generated API client:

- `pnpm generate:api` — Fetch OpenAPI spec and regenerate the client

---

## ✅ PR checklist (Frontend)

- Client-first: server components only for SEO or extreme TTFB needs
- Uses generated API hooks; no new `BackendAPI` usages
- UI uses `src/components` primitives; no new `_legacy__` components
- Logic is separated into `use*.ts` and `helpers.ts` when non-trivial
- Reusable logic extracted to `src/services/` or `src/lib/utils.ts` when appropriate
- Navigation uses the Next.js router
- Lint, format, type-check, and tests pass locally
- Stories updated/added if UI changed; verified in Storybook

---

## ♻️ Migration guidance

When touching legacy code:

- Replace usages of `src/components/_legacy__/*` with the modern design system components under `src/components`
- Replace `BackendAPI` or `src/lib/autogpt-server-api/*` with generated API hooks
- Move presentational logic into render files and data/behavior into hooks
- Keep one-off transformations in local `helpers.ts`; move reusable logic to `src/services/` or `src/lib/utils.ts`

---

## 📚 References

- Design system (Chromatic): [`https://dev--670f94474adee5e32c896b98.chromatic.com/`](https://dev--670f94474adee5e32c896b98.chromatic.com/)
- Project README for setup and API client examples: `autogpt_platform/frontend/README.md`
- Conventional Commits: [conventionalcommits.org](https://www.conventionalcommits.org/)
