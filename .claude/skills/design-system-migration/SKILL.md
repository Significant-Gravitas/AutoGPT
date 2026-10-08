---
name: design-system-migration
description: >-
  Moving an existing component library over to Kobra, one screen at a time.
  Use when a project on shadcn/ui, Radix, Material UI, Chakra, Mantine, Ant
  Design, Headless UI, Bootstrap or a homegrown component folder wants to
  adopt Kobra components; when two libraries have to coexist during a
  migration; when mapping an old theme to Kobra's tokens; when translating
  Radix or other headless APIs to Base UI; when installing from the @kobra
  shadcn registry; or when planning, tracking, reviewing and finishing a
  migration without a big-bang rewrite. Has a plan mode, a build mode and a
  review mode.
---

# Design System Migration

A migration that starts with "replace the button everywhere" ends with two
buttons, three months later, and a team that has stopped believing the second
one is coming. The way through is to move the foundations first, then move
whole screens, and to let the old library live until the last screen has
left it. Nothing is half-migrated; a screen is on the old system or on the
new one.

This skill is the method for doing that with Kobra: the order to do things
in, how to install, how to translate, how to keep both systems on one page
without them fighting, and what "done" means for a screen.

## The order

1. **Inventory.** Count what the project uses before deciding anything. See
   [inventory.md](inventory.md).
2. **Foundations.** Tokens, fonts, radius, dark mode, the `cn` helper, the
   registry, the sound layer. All of it lands before a single component
   changes, so every screen that follows is built on the final ground. See
   [tokens.md](tokens.md).
3. **Primitives.** The leaf components every screen uses: Button, Input,
   Checkbox, Switch, Select, Badge, Tooltip. Installed from the registry,
   aliased so old import paths keep resolving.
4. **Screens.** One at a time, in order of traffic, each one moved whole and
   reviewed before the next begins. See [screens.md](screens.md).
5. **Retire.** Delete the old package, the compat layer and the aliases.
   Verify with the bundle, not with a grep.

Never skip from 1 to 4. A screen migrated onto tokens that later change is a
screen migrated twice.

## The rules that hold

**Install, do not copy.** Every Kobra component arrives through the
registry, so it can be updated and its dependencies are installed with it.
The project's `components.json` carries the namespace and the token; the
command is one line per component:

```bash
npx shadcn@latest add @kobra/button
```

**Whole screens, not whole components.** Replacing the button everywhere
touches every screen and finishes none. Replacing one screen's buttons,
inputs and dialogs finishes that screen and can be reviewed as a unit.

**Alias, then delete.** Old import paths keep working through the migration
by pointing at the new components. The alias file is the list of what is
left to do; when it is empty, the migration is over.

**Base UI is not Radix with a different name.** `asChild` becomes `render`,
`data-state` becomes `data-open` and friends, positioning parts are named
differently, and transforms have their own origin variable. Translate on
purpose. See [primitives.md](primitives.md).

**Tokens are the contract.** The old palette maps onto Kobra's semantic
tokens once, in the stylesheet. No component carries a hex value.

**Both systems in one document, neither in charge.** Scope the old
library's global styles, keep Tailwind's preflight authoritative, and use
`@layer` so utilities win where they should. See [tokens.md](tokens.md).

**A screen is done when it is reviewed.** Visual diff against the old
screen, keyboard walk, every state, dark mode, reduced motion, a narrow
viewport. Then the old imports on that screen are gone and the alias count
went down.

**Keep a ledger.** A `MIGRATION.md` at the root with the component map, the
screen list with status, and the decisions made. It is the thing a new
teammate reads, and it is what makes the migration resumable after a break.

## Planning a migration

When asked to plan, produce:

1. The inventory table: old component, usage count, Kobra target, notes.
2. The foundations list, with what changes in the stylesheet.
3. The screen order, by traffic, with the components each one needs.
4. Risks: components with no target, custom behavior to preserve, places
   where the two systems will share a page for a while.
5. The first pull request, scoped to foundations only.

## References

| File                           | Read it for                                                           |
| ------------------------------ | --------------------------------------------------------------------- |
| [inventory.md](inventory.md)   | Counting usage, the component map from common libraries to Kobra      |
| [tokens.md](tokens.md)         | Theme mapping, fonts, radius, dark mode, coexistence, the sound layer |
| [primitives.md](primitives.md) | Radix and other headless APIs translated to Base UI                   |
| [screens.md](screens.md)       | The per-screen playbook, the definition of done, the ledger format    |
| [review.md](review.md)         | Reviewing a migrated screen and the required output format            |

## Reviewing

When asked to review a migrated screen or a migration in progress, follow
[review.md](review.md): compare against the old screen, walk every state and
input method, report findings in one table with Severity, Location, Now,
Change and Why columns, list what was left alone and what was verified, and
end with a verdict.
