---
name: emil-unslop-code
description: "Strip the tells that mark code as AI-generated — narrated comments, try-catch theater, silent fallbacks, dead scaffolding, `as any` — then make the diff read like the codebase it lands in. Use when cleaning up agent-written code, before opening a PR, when a diff \"looks AI-generated\", or as a pass after any generated change. Triggers on: code slop, AI slop, slop, de-slop, unslop, clean up this code, obvious comments, redundant comments, narrated comments, commented-out code, stale comments, try-catch, swallowed error, silent failure, fallback, defensive, null checks, as any, ts-ignore, non-null assertion, console.log, debug logs, TODO, dead code, unused imports, duplicate helper, one-call wrapper, V2, enhanced, legacy alias, backwards compatibility, minimal diff, drive-by changes, tautological test, skipped test, \"reads like Copilot wrote it\"."
---

# Unslop Code

Nobody flags AI code for one bad line. They recognize the accumulation: a comment narrating every statement, a try-catch around code that can't throw, a fallback to an empty array, a helper that already exists two files over. Each choice is defensible alone; together they're a watermark. The cause is mechanical: models were trained where code that *runs* wins, so they over-defend, over-comment, and over-scaffold — everything that makes a snippet survive in isolation makes it slop inside a codebase.

Slop is rarely buggy. It compiles, it passes tests, and it still costs you: it buries intent, hides real failures behind quiet fallbacks, and doubles the surface the next reader has to hold in their head. This skill is a pass over any generated change, and the default fix for almost every pattern in it is deletion, not rewriting.

## The pass

1. Scan the diff (or file) for the patterns below, in order: comments, defense, types, naming, structure, logging, tests, scope.
2. Fix by deleting first. Rewrite only where behavior must survive. When removing a cast, catch, or fallback forces a choice about what happens on bad input, choose the loud failure — a throw or a failed build — over the quiet wrong answer.
3. Match the room: mirror the surrounding codebase's comment density, naming, and error-handling idiom. Slop is relative — a comment style that's normal in one repo is noise in another.
4. Final read: "What here would make a reviewer suspect a model wrote this?" Fix it. Repeat until the answer is nothing.

## Comments

Comments carry the strongest signal, and one test covers all of them: **does this comment state something the code cannot?** A constraint, a unit, a tradeoff, the bug it works around — keep. Everything else — delete.

### Comments that narrate

`// increment the counter` above `count++`. `// Loop through each item`. `// Return the result`. The model comments the way it was rewarded to in training: constantly. Narration restates the next line and doubles the code's length for zero information. Delete every one.

### Comments that talk to the reviewer

`// Updated to use the new API`. `// Removed the old implementation`. `// Now handles null`. `// NEW:`, `// Fixed:`. These describe the *diff*, not the code — and the moment the change merges they describe nothing, because the next reader never saw the old version. That explanation belongs in the commit message. Delete.

### Comments that lie

A comment describing behavior the code no longer has, a parameter that was removed, a step that moved elsewhere. Worse than no comment, because a reader trusts it over the code. When editing any region, re-read every comment in it and delete or correct the ones the edit invalidated.

### Section dividers and numbered steps

`// ===== Helpers =====`, `// Step 1: parse input`. A function that needs numbered steps needs extracted functions with those names instead. Delete the divider, or do the extraction.

### Docstrings that restate the signature

`@param userId - the ID of the user`. Bytes, not information. Document constraints, units, side effects, and surprising behavior — or document nothing.

### Commented-out code and summary blocks

Delete commented-out code; git has it, and "kept just in case" is what version control is for. Delete file-level blocks that list everything the code below does; they go stale on the first edit and nobody reads them.

## Defensive theater

Models over-defend because defensive code "succeeds" more often. In a real codebase the defense inverts: it hides bugs instead of preventing them.

### Checks against impossible states

A null check on a value the type system, the constructor, or the three lines above already guarantee. `if (!user) return;` after a call that throws when user is missing. Trust the invariant — and if it genuinely isn't guaranteed, fix the type once instead of checking at every call site.

### Catch theater

Catching only to log and rethrow unchanged. An empty catch. `catch (e) { console.error(e) }` that swallows the failure and continues as if it succeeded. Handle errors once, at a real boundary, adding context — or don't catch at all. A stack trace beats a log line plus corrupted state.

### Silent fallbacks

`catch { return [] }`. `?? defaultValue` masking a failed fetch. A simpler backup code path for when the real one fails. This is the most dangerous pattern on the list, because the fallback keeps tests green while the primary path is broken — you can no longer tell what you're actually running. Let it fail loudly, unless degraded behavior is an explicit product decision, and then make the degraded state visible to the user.

### Validation at every layer

The same null-and-range check repeated in the handler, the service, and the helper. Validate untrusted input once at the boundary, then pass a trusted value through — repetition just means three places to update when the rule changes.

### Condition padding

`x !== null && x !== undefined && x !== ''` where one check suffices. `=== true` on a boolean. `a?.b?.c` chains on values guaranteed to exist. Write the check the invariant actually needs, not the one that looks safest.

## Type escape hatches

`as any`, `as unknown as X`, `@ts-ignore`, a non-null `!` used to silence an error instead of fixing it. The compiler error was information; the cast deletes it and moves the failure to runtime, where it lands on a user. Parse or validate at the boundary and let the correct type flow through. A cast survives the pass only with a comment stating the invariant that makes it safe.

## Naming

### Generic names

`data`, `result`, `item`, `temp`, `handler`, `processData`. The model names what the value *is structurally*; name what it *means*: `unpaidInvoices`, `retryDelayMs`. If you can't name the domain fact, you haven't understood the line — which is the thing worth fixing.

### Version-suffixed names

`fetchDataV2`, `handleClickNew`, `EnhancedButton`, `utils-new.ts`. These exist because the old thing was kept alongside the new one. Replace the old implementation, update its callers, and take its name. The one exception is published public API — an npm package's exports, a served endpoint — where callers exist you can't update; keep a deprecated alias there and only there.

## Structure

### One-call wrappers

A helper that only forwards its arguments to another function is indirection with no policy. Inline it; extract again when there's a second caller or real logic to own.

### Speculative flexibility

An options object with one caller passing one value. An interface with a single implementation. Config flags for scenarios the task doesn't have. Every unused degree of freedom is something the next reader must rule out before they understand the code. Build for the current caller; generalize on the second real use.

### Duplicate helpers

A reimplemented formatter, validator, or permission check that already exists — the model didn't search the repo, so it wrote its own. Before adding any utility, search for the existing owner of that job.

### Unrequested compatibility shims

Deprecated aliases, re-exports, "keeping the old signature just in case" when every caller lives in this repo. Update the callers instead — that's what makes it one repo.

### Dead scaffolding

Helpers from an approach that was abandoned mid-task. Unused imports and variables. `TODO: implement error handling` stubs. After the solution works, do a minimum-patch pass: everything not load-bearing for the final approach gets deleted.

## Logging

`console.log('here')` and entry/exit logs (`entering processOrder`) are debug leftovers — delete before finishing. `console.log('✅ Successfully initialized!')` is ceremony; emoji and celebration in log output is one of the strongest single tells. Log the facts an operator needs, in the repo's existing log style, or log nothing.

## Tests

### Never delete a test to pass it

Deleting, skipping, or weakening an assertion to make a run green destroys the report of the bug. Fix the implementation — or, if the contract genuinely changed, say so explicitly and update the test to assert the new contract.

### Tautological tests

Reproducing the implementation's calculation inside the test and comparing the two passes even when both are wrong. Assert an independently known outcome, at the public boundary.

## Scope

Reformatting, renaming, or refactoring code the task didn't touch buries the real change in the diff and makes review slower and riskier. Keep the patch minimal; propose unrelated cleanup as its own change.

## Final audit

- [ ] Every surviving comment states something the code cannot
- [ ] No catch swallows a failure; no fallback hides a broken primary path
- [ ] No cast silences the compiler without a stated invariant
- [ ] No name carries `V2`, `New`, or `Enhanced`; the old implementation is gone
- [ ] No helper duplicates an existing one; no wrapper just forwards
- [ ] No debug logs, no dead code, no TODO stubs, no commented-out blocks
- [ ] No test was weakened to pass
- [ ] The diff touches only what the task required
- [ ] The result reads like the surrounding codebase wrote it
