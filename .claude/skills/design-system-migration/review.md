# Review

How to review a migrated screen, or a migration in progress, and the shape
the report takes.

## Protocol

1. Read the ledger. Confirm the screen under review is listed and its
   status matches the PR.
2. Grep the screen's files for old-library and compat imports. Any hit is
   a finding.
3. Open old and new side by side at the same viewport. Walk: populated,
   empty, loading, error, overflow, long labels, narrow viewport, dark mode,
   reduced motion.
4. Walk the keyboard: tab order, focus visible, Escape, arrow keys.
5. Check custom behavior from the inventory against the new screen.
6. Turn the sound layer on and press every control; a silent one is a
   leftover.
7. Check that removed CSS was actually removed and that nothing else
   depended on it.
8. Record what was actually done.

## Severity

- `HIGH`: an old-library import remains on a "done" screen; a keyboard path
  that worked before and does not now; custom behavior silently dropped; a
  state that renders wrong; a hex color in a component instead of a token.
- `MEDIUM`: a compat import where a direct one belongs; an intended visual
  change not listed in the PR; a token mapped inconsistently with the
  ledger; icons from two sets on one surface; a busy button that rewrites
  its label.
- `LOW`: polish on the migrated screen that the old one also lacked.

## Output

### Scope

One paragraph: screen or screens, the old library, the branch, what was
compared against. Then coverage:

| Area     | What was inspected                       | Result                                      |
| -------- | ---------------------------------------- | ------------------------------------------- |
| Imports  | files grepped, hits                      | n findings, `Clear`, or `Not reviewed: why` |
| States   | which were walked, at which viewports    |                                             |
| Keyboard | paths walked                             |                                             |
| Behavior | inventory items checked                  |                                             |
| Tokens   | colors, radius, fonts against the ledger |                                             |
| Ledger   | row present, status accurate             |                                             |

### Findings

| Severity | Location                   | Now                                    | Change                                           | Why                                                 |
| -------- | -------------------------- | -------------------------------------- | ------------------------------------------------ | --------------------------------------------------- |
| HIGH     | `src/routes/inbox.tsx:12`  | `import { Chip } from '@mui/material'` | `Badge` from `@/components/ui/badge`             | The screen is marked done with an old import on it  |
| MEDIUM   | `src/routes/inbox.tsx:88`  | `{pending ? 'Saving…' : 'Save'}`       | `<BusySpinner busy={pending}>Save</BusySpinner>` | The label is rewritten into a status while it works |
| LOW      | `src/routes/inbox.tsx:140` | `rounded-xl` inside `rounded-xl p-2`   | Outer `rounded-2xl`                              | Nested corners are not concentric                   |

### Left alone

| Location                  | Candidate                          | Left because                                                                      |
| ------------------------- | ---------------------------------- | --------------------------------------------------------------------------------- |
| `src/routes/inbox.tsx:60` | Replace the hand-written typeahead | It is on the ledger as behavior to carry; Base UI's menu covers it in the next PR |

### Verified

What was opened, compared, walked and pressed. Anything skipped is **Not
verified**.

### Verdict

`Block` while any `HIGH` stands, `Needs changes` for `MEDIUM` or `LOW`,
`Approve` only when nothing actionable remains. List unverified checks
beside it.
