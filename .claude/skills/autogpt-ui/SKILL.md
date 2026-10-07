---
name: autogpt-ui
description: "Build or change AutoGPT Platform frontend UI on the design system: read DESIGN.md, compose atoms, molecules and organisms, use palette and semantic classes, lint every touched file, and add a story for design-system components. TRIGGER when writing or editing .tsx under autogpt_platform/frontend/src that renders UI, adding a component to src/components, or restyling a page."
user-invocable: true
args: "[what to build or change] — optional; the skill applies to any frontend UI work."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Build UI on the AutoGPT design system

All paths are relative to `autogpt_platform/frontend`.

## 1. Read the reference first

Read `DESIGN.md` before writing UI. It has the decisions (zinc primary, purple-500 accent, `#FAFAFA` page background, 12px fields, 16px cards, no `dark:`), the tokens, the component catalog, and the rules with the lint rule behind each. When it and the code disagree, the code in `src/components/styles/colors.ts`, `src/app/globals.css`, `tailwind.config.ts` and `src/components/atoms/Text/helpers.ts` wins.

## 2. Pick components from the catalog

Look up what you need in the catalog section of `DESIGN.md`, then open the component and its `*.stories.tsx` to see its props.

- Typography: `Text` (`variant`, `tone`, `as`). Never a raw `<p>`/`<h*>` with classes. Pass `unmask={false}` when it renders user data.
- Actions: `Button` (`variant`, `size`, `as="NextLink"` for a link that looks like a button). Never a raw `<button className>`.
- Links: the `Link` atom. Do not import `next/link` in feature code.
- Icons: `<Icon icon={SomeIcon} size={16} />` with data from `@hugeicons/core-free-icons`. No other icon library.
- Fields: `Input` (text, password, number, amount, textarea), `Select`, `DateInput`, `TimeInput`, `FileInput`, `SearchInput`, `Switch`.
- Overlays and structure: `Dialog`, `Popover`, `DropdownMenu`, `Tooltip`, `TabsLine`, `Collapsible`, `Card`, `Table`, `Badge`, `Alert`, `Skeleton`, `ErrorCard`.
- Toasts: `useToast` / `toast` from `molecules/Toast/use-toast`, never `sonner`.

Never import from `src/components/__legacy__` or, outside `src/components`, from `src/components/ui`. If nothing in the catalog fits, say so and compose from atoms; do not copy a legacy component.

## 3. Style with tokens

- Colours: semantic classes first (`bg-background`, `text-muted-foreground`, `border-border`, `bg-card`), otherwise the palette families `zinc`, `slate`, `red`, `orange`, `yellow`, `green`, `purple`, `pink`.
- Banned by lint: `gray`, `neutral`, `stone`, `emerald`, `amber`, `violet`, `indigo`, `rose`, `lime`, `fuchsia`. Map gray/neutral/stone to `zinc`, emerald to `green`, amber to `yellow`, violet/indigo to `purple`, rose to `red`. Avoid new `blue`, `sky`, `teal` and `cyan` (no decision yet).
- No `dark:` classes, no hex or `rgb()` colour classes, no `!important` overrides of an atom (fix or extend the atom instead), no arbitrary values that the scale already has (`h-[2.25rem]` is `h-9`).
- Text colour comes from `Text`'s `tone`, not `!text-zinc-*`.
- Keyboard handlers use `isKey(e, "Enter")` from `@/lib/keyboard`.

## 4. Lint every file you touch

From `autogpt_platform/frontend`:

```bash
npx eslint --fix <changed files>
npx prettier --write <changed files>
```

In Claude Code the repo's `.claude/settings.json` hook already runs `npx eslint --fix` after each edit under `src/` and prints what is left; fix what it reports, do not suppress it. Never add a file to `eslint-allowlist.json`. If you removed the last violation from an allowlisted file, delete its entry (or run `npx tsx scripts/eslint-allowlist-regenerate.ts`); `scripts/eslint-allowlist.test.ts` fails on stale entries.

## 5. Add a story for design-system components

Any component you add or change under `src/components/{atoms,molecules,organisms}` gets a `ComponentName.stories.tsx` next to it:

```tsx
import type { Meta, StoryObj } from "@storybook/nextjs";
import { MyAtom } from "./MyAtom";

const meta = {
  title: "Atoms/MyAtom",
  component: MyAtom,
  tags: ["autodocs"],
  parameters: { layout: "centered", a11y: { test: "error" } },
  args: {},
} satisfies Meta<typeof MyAtom>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
```

Add one story per variant and state (disabled, loading, error, empty). Data-fetching components mock the API with the generated MSW handlers in `parameters.msw.handlers`. Check it with `pnpm storybook`, and add the component to the catalog in `DESIGN.md`.

## 6. Before you finish

Run `pnpm format`, `pnpm lint`, `pnpm types` and the Vitest files near your change (`npx vitest run <dir>`). Fill in the design-system checklist in the PR template.
