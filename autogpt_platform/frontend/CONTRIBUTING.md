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

## 📟 Contribution process

### 1) Branch off `dev`

- Branch from `dev` for features and fixes
- Keep PRs focused (aim for one ticket per PR)
- Use conventional commit messages with a scope (e.g., `feat(frontend): add X`)

### 2) Feature flags

If a feature will ship across multiple PRs, guard it with a flag so we can merge iteratively.

- Use LaunchDarkly-based flags (see Feature Flags below)
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
- Use [route segments](https://nextjs.org/docs/app/building-your-application/routing) and server actions as needed; no `pages/`

### Client-first policy

- Default to client components
- Use server components only when:
  - SEO requires server-rendered HTML, or
  - Extreme first-byte performance justifies it
- If you render server-side data, prefer server-side prefetch + client hydration (see examples below and [React Query SSR & Hydration](https://tanstack.com/query/latest/docs/framework/react/guides/ssr))
- Prefer using [Next.js API routes](https://nextjs.org/docs/pages/building-your-application/routing/api-routes) when possible over [server actions](https://nextjs.org/docs/14/app/building-your-application/data-fetching/server-actions-and-mutations)

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

### Styling and components

- [Tailwind CSS](https://tailwindcss.com/docs) + [shadcn/ui](https://ui.shadcn.com/) ([Radix Primitives](https://www.radix-ui.com/docs/primitives/overview/introduction) under the hood)
- Use the design system under `src/components` for primitives and building blocks
- Do not use anything under `src/components/_legacy__`; migrate away from it when touching old code
- Reference the design system catalog on Chromatic: [`https://dev--670f94474adee5e32c896b98.chromatic.com/`](https://dev--670f94474adee5e32c896b98.chromatic.com/)
- Use the [`tailwind-scrollbar`](https://www.npmjs.com/package/tailwind-scrollbar) plugin utilities for scrollbar styling

---

## 🧱 Component structure

For components, separate render logic from data/behavior, and keep implementation details local.

Suggested structure when a component has non-trivial logic:

```
FeatureX/
  FeatureX.tsx        (render logic only)
  useFeatureX.ts      (hook; data fetching, behavior, state)
  helpers.ts          (pure helpers used by the hook)
  components/         (optional, subcomponents local to FeatureX)
```

Guidelines:

- Prefer function declarations for components and handlers
- Only use arrow functions for small inline lambdas (e.g., in `map`)
- Avoid barrel files and `index.ts` re-exports
- Keep component files focused and readable; push complex logic to `helpers.ts`
- Abstract reusable, cross-feature logic into `src/services/` or `src/lib/utils.ts` as appropriate

---

## 🔄 Data fetching patterns

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
- Component props should be `interface Props { ... }`
- Use precise types; avoid `any` and unsafe casts

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

- End-to-end: [Playwright](https://playwright.dev/docs/intro) (`pnpm test`, `pnpm test-ui`)
- [Storybook](https://storybook.js.org/docs) for isolated UI development (`pnpm storybook` / `pnpm build-storybook`)
- For Storybook tests in CI, see [`@storybook/test-runner`](https://storybook.js.org/docs/writing-tests/test-runner) (`test-storybook:ci`)
- When changing components in `src/components`, update or add stories and visually verify in Storybook/Chromatic

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
