# Lane E — Stability & States

The lane the seeded dev account hides. Every data-driven view has four states (loading, empty, error, full) and the person who built it only ever saw the last one. A new user sees the other three first. This lane also covers everything that makes the page move after the user has started reading it. Owning skills: `emil-ui-polish` (stability, stacking) and `emil-design-foundations` (what the states should say).

For each view in the recon's data-driven inventory, find the four branches in the code. A branch that doesn't exist is the finding.

## E1. Empty states exist and teach — BLOCKER on a first-run screen, SHOULD FIX elsewhere

An empty list is a first-run screen, not an error. A view that renders `null`, a bare table header, or "No items" when there's no data is the first thing every new account sees.

**Hunt for:** `.map()` with no `length === 0` branch; `return null` on empty data; copy like "No items", "No data", "Nothing here", "No results found." with nothing after it.

The shape: a short headline, one line of explanation, and the primary action that creates the first item.

```jsx
if (projects.length === 0) {
  return (
    <EmptyState
      title="No projects yet"
      description="Projects hold your team's work. Create one to get started."
      action={<Button onClick={createProject}>Create project</Button>}
    />
  );
}
```

Size the empty state to the dimensions of the filled state, so completing the action doesn't shift the layout. A search or filter that returns nothing is a different empty state: say what was searched and offer to clear it.

The branch is a fix. The words are content: if the project has no empty-state copy to follow, write the branch with the closest existing pattern and flag the copy for the human list (Hard Rule 8).

## E2. Failed requests render something useful — BLOCKER

The network will fail on launch day. A view that renders nothing, spins forever, or crashes to a white screen when its request fails is broken for everyone on a train.

**Hunt for:** data hooks whose `error` / `isError` is never read; `fetch` with no `.ok` check; routes with no `error.tsx` / error boundary / `errorElement`; no `not-found` page; "Something went wrong" with no next step.

An error state says what happened and offers a specific recovery: a retry button that refetches, a link back to somewhere that works. "Something went wrong" plus a back button ends the journey. The framework's default error and 404 pages are findings; they're unstyled, off-brand, and in production they say nothing.

## E3. Every wait over 400ms has an indicator — SHOULD FIX

Silence past 400ms reads as broken. The user clicks again (see D1) or leaves.

**Hunt for:** data hooks whose `isLoading` / `isPending` is never read; route segments with no `loading.tsx` or `<Suspense fallback>`; buttons that start async work with no pending state; `fallback={null}`.

The opposite is also a finding: a spinner that flashes for 80ms on a fast response. Delay the indicator ~150ms so fast responses never show it.

## E4. Loading states hold the page's shape — SHOULD FIX

A spinner collapses the page to nothing, then the content arrives and everything jumps. A skeleton occupies the same box the loaded content will, so nothing moves when the data lands.

**Hunt for:** a centered spinner as the loading state of a list, table, card grid, or page; skeletons with a different height or row count than the loaded content; `dynamic()` / `lazy()` imports with no sized fallback.

```css
.row-skeleton { height: 48px; }   /* the loaded row's height, not a guess */
```

Spinners are right for an action inside a control (a submitting button). Skeletons are right for content.

## E5. Persisted state is correct on first paint — SHOULD FIX

The theme that flashes light before going dark, the sidebar that renders open and then snaps shut, the tab that resets and then jumps to the saved one. Each is a value read in `useEffect` *after* the first paint.

**Hunt for:** `useEffect` that reads `localStorage` and then calls a state setter for theme, sidebar, layout, tab, or density; `dark:` classes applied by a client-only hook; `useState(false)` followed by an effect that corrects it.

Fix by where the state lives:

- **Theme**: a blocking inline script in `<head>` that sets the class or `data-theme` before paint. `next-themes` does this; the finding is a hand-rolled toggle.
- **Server-rendered layout state** (sidebar open, density): store it in a cookie and read it on the server, so the HTML arrives correct.
- **Client-only state**: `useState(() => readStoredValue())` with a lazy initializer, and render nothing size-dependent until it's known.

## E6. Changing numbers and labels don't shift the layout — SHOULD FIX

Proportional digits have different widths, so a ticking counter or a price that updates jitters everything beside it. A button whose label changes from "Save" to "Saving…" changes width and shoves its neighbors.

**Hunt for:** counters, timers, prices, percentages, table number columns, and pagination without `tabular-nums`; buttons with swapping labels and no fixed or minimum width.

```css
.counter, .price, td.numeric { font-variant-numeric: tabular-nums; }
```

Tailwind: `tabular-nums`.

## E7. Font weight never changes on interaction — SHOULD FIX

Bold-on-hover or bold-on-selected reflows the text and shifts every sibling by a pixel or two. Keep the weight constant and signal state with color or background.

**Hunt for:** `font-weight` inside `:hover`, `.active`, `.selected`, `[aria-selected]`, `[data-state="active"]`; `hover:font-medium`, `data-[state=active]:font-semibold`.

```css
/* Bad */
.tab:hover { font-weight: 600; }

/* Good */
.tab { font-weight: 500; }
.tab[aria-selected="true"] { color: var(--gray-12); }
```

## E8. Late UI doesn't push content — SHOULD FIX

Anything that mounts after load and takes up space in the flow moves the page under the user: a cookie banner, an announcement bar, an "install the app" prompt, a verify-your-email strip.

**Hunt for:** banners rendered conditionally at the top of the layout after a client-side check.

Fix: overlay it (`position: fixed`, with safe-area padding per C8), or reserve its space from the first paint by deciding on the server whether it shows.

## E9. Stacking comes from a scale — SHOULD FIX

`z-index: 9999` is how a toast ends up behind a modal, a dropdown gets clipped under a sticky header, and a tooltip disappears inside a dialog. These only show up when two layers meet, which is never on the screen the developer was looking at.

**Hunt for:** `z-index` values over 100 that aren't tokens; `z-[9999]`, `z-50` sprinkled without a system; portaled and non-portaled overlays mixed.

```css
:root {
  --z-dropdown: 100;
  --z-modal: 200;
  --z-tooltip: 300;
  --z-toast: 400;
}
```

If the project has a scale, move the strays onto it. If it doesn't, don't invent one mid-sweep: fix the collisions that are provably wrong (toast under modal) and note the rest. Where a component only needs its children contained, `isolation: isolate` beats a z-index. Put on the device list: "open a modal, trigger a toast, open a dropdown inside the modal, hover a tooltip inside the dropdown".

## E10. Long content is handled — SHOULD FIX

The dev data had names like "Test Project". Real data has a 90-character project name, an email with no break opportunity, and a user whose name is in German.

**Hunt for:** cards, table cells, sidebar items, breadcrumbs, and avatars-with-names that render user-generated strings with no `truncate`, `line-clamp`, `min-width: 0`, or `overflow-wrap`.

```css
.cell-text {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.card-text { overflow-wrap: break-word; }
```

A flex child needs `min-width: 0` before `text-overflow: ellipsis` does anything. Truncation needs an escape hatch: a `title` attribute, a tooltip, or an expand control. An ellipsis with no way to read the rest hides data.

## E11. Decorative layers don't steal clicks — SHOULD FIX

A gradient overlay, a glow, or an illustration positioned over a control swallows its clicks. It works on the developer's viewport and fails at the width where the two overlap.

**Hunt for:** absolutely positioned decorative elements (gradients, noise, glows, fades) without `pointer-events: none`.

```css
.hero-glow, .fade-overlay { pointer-events: none; }
```

Code-built illustrations also get `user-select: none`.

## E12. Optimistic updates revert on failure — SHOULD FIX

An optimistic update that never reverts shows the user a lie: the item looks saved, the toggle looks on, and the server never heard about it.

**Hunt for:** state set before an `await` with no `catch` restoring the previous value; `onMutate` with no `onError` rollback (React Query); `useOptimistic` with no error surface.

Fix: on failure, restore the previous value and say so ("Couldn't update. Try again.").

## E13. Copy actions confirm — POLISH

A copy-to-clipboard button with no feedback gets clicked three more times. Swap to a checkmark for ~1.5s.

## E14. Walk the app as a new user — always

Put on the device list: "sign up with a fresh account and screenshot every screen before adding any data", "open DevTools → Network → Offline, then click through the main views", "throttle to Slow 3G and watch what each screen shows while it waits", "hard-refresh on every route with dark mode and a collapsed sidebar saved". These are the states nobody has looked at. Report them as checks to run, never as verified.
