# Screens

The unit of migration is a screen. Here is what happens to one.

## Order

By traffic, highest first, with one exception: the first screen migrated
is a small one with common components, so the process is learned on
something forgiving. After that, the screens people see most, because that
is where the old system's presence is most expensive.

Group screens that share a layout shell; the shell moves with the first of
them.

## The playbook for one screen

1. **Read it.** List the components it uses, the states it has, the
   custom behavior in it. Open it in the old build and keep it open.
2. **Branch.** One branch per screen. The PR title is the screen's name.
3. **Replace, top down.** Layout shell, then containers, then controls.
   Import from `@/components/ui` directly, not from the compat layer.
4. **Carry behavior.** Every item from the inventory's custom-behavior
   list that belongs to this screen is reimplemented on the new component
   or deliberately dropped, with the ledger saying which.
5. **Remove the old CSS.** Any stylesheet or styled-components file that
   only this screen used goes with it.
6. **Walk every state.** Empty, loading, error, populated, overflow, a
   long label, a narrow viewport, dark mode, reduced motion.
7. **Walk the keyboard.** Tab order, focus visible, Escape closes what it
   should, arrow keys where lists are.
8. **Diff against the old.** Side by side, at the same viewport. Differences
   are either intended (and listed in the PR) or fixed.
9. **Listen.** With the sound layer on, every control on the screen
   answers. A silent control is an old one that was missed.
10. **Update the ledger.** Status to done, remaining old imports to zero
    for this screen, compat files that are now unused deleted.

## Definition of done

A screen is done when all of these are true:

- No import from the old library or the compat layer remains in its files.
- Every state was looked at, and the ones that changed on purpose are
  listed in the PR.
- Keyboard and focus were walked.
- Dark mode and a narrow viewport were looked at.
- The visual diff was reviewed by someone other than the person who did it.
- The ledger row is updated.

Not done: "the components are swapped and it looks the same." Looking the
same is step 8 of 10.

## The ledger

`MIGRATION.md` at the repository root:

```markdown
# Migration to Kobra

Started 2026-09-10. Old library: @mui/material 5.

## Foundations

- Tokens mapped (see below). Radius 8px kept from MUI.
- Dark mode: `.dark` set alongside MUI's `data-theme` until MUI is gone.
- Sound layer installed, wrapped in `app/root.tsx`.

## Component map

| Old | Files | Target | Decision | Notes |
| --- | ----- | ------ | -------- | ----- |

## Screens

| Screen   | Components                         | Status      | PR   | Notes                                              |
| -------- | ---------------------------------- | ----------- | ---- | -------------------------------------------------- |
| Settings | Input, Switch, Button, Dialog      | done        | #412 | Dropped the unsaved-changes prompt; it never fired |
| Inbox    | Table, Badge, Dropdown Menu, Sheet | in progress | #418 |                                                    |

## Remainders

| Component | Where         | Owner | Plan                     |
| --------- | ------------- | ----- | ------------------------ |
| Rating    | feedback page | —     | drop with the page in Q4 |

## Decisions

- Tabs replace MUI's Stepper on the onboarding flow; progress shown under.
```

It is plain markdown so anyone can edit it in a PR, and it is the only
place status lives. A migration tracked in three tools is tracked in none.

## Cadence

One screen per PR, small enough to review in a sitting. Foundations PR
first, then screens, then a final PR that removes the old dependency. Do
not batch screens to save review time; the review is the point.

## Retiring the old library

When the ledger shows every screen done:

1. Delete the compat directory. The build fails on any import that was
   missed; fix each by migrating properly, not by restoring the file.
2. Remove the old package. Build. Grep the output bundle for the old
   library's class prefix or a distinctive string; it should not be there.
3. Remove the old theme provider, its reset and its fonts.
4. Remove the dark-mode bridge.
5. Close the ledger with the date and leave it in the repository. It is the
   record of why things are the way they are.
