---
name: emil-engineering-vocabulary
description: Reverse-lookup glossary that turns a vague description of a frontend engineering concept into its exact term ("the search waits until I stop typing" → Debounce; "the UI updates before the server confirms" → Optimistic update). Use when the user asks "what's it called when…", or describes a behavior, pattern, or browser quirk without knowing its name and wants the right word to prompt an AI or engineer with. For naming a concept, not implementing one.
---

# Engineering Vocabulary

Turn a vague description of a frontend engineering concept into the precise term, so the user knows what to ask for.

## Quick Start

The user describes a concept loosely. You return the matching term(s) in this format:

```
**Debounce** — Wait until the events stop (typing, resizing) before running the handler, so it fires once at the end instead of on every event.
```

If several terms could fit, list the best match first, then 1–2 alternates with a one-line note on how they differ.

## Instructions

1. **Read for intent, not keywords.** Users describe symptoms and behavior ("it fires too often", "the page jumps", "my z-index doesn't work"), not the technical name. Map the symptom to the glossary below.
2. **Quote the glossary verbatim.** Its descriptions are authoritative — use them as-is, don't paraphrase.
3. **Disambiguate close terms.** When two compete (*Debounce* vs *Throttle*, *Lazy loading* vs *Code splitting*, *SSR* vs *SSG*, *Controlled* vs *Uncontrolled*), contrast them so the user can pick.
4. **When nothing matches exactly,** name the closest term and say plainly it's an approximation, or describe the concept in the glossary's vocabulary (e.g. "that's a *race condition* between two *optimistic updates*").
5. **Stay within this glossary.** If a term genuinely isn't here, say so rather than inventing one — though you may explain the concept using these words.
6. **Keep it tight.** A naming question wants a name, not an essay. Lead with the term; expand only if asked.

## Examples

**Example 1 — symptom-based**
User: "What's it called when the search input waits until I stop typing before it actually searches?"
Output:
```
**Debounce** — Wait until the events stop (typing, resizing) before running the handler, so it fires once at the end instead of on every event.

Close alternate:
- **Throttle** — if it runs *during* the activity, just capped to once per interval.
```

**Example 2 — behavior-based**
User: "The thing where the like button updates instantly even though the request hasn't finished."
Output:
```
**Optimistic update** — The UI updates immediately as if the request already succeeded, then rolls back if it fails.
```

**Example 3 — browser quirk**
User: "My dropdown has z-index 9999 but it still renders behind the header."
Output:
```
**Stacking context** — A group whose children's z-index values only compete with each other, not the rest of the page — why a huge z-index can still lose to an element outside the group.
```

## Glossary

### Rendering & Loading — how a page gets to the screen
- **Server-side rendering (SSR)** — The server sends fully-built HTML for each request, so content shows before JavaScript loads.
- **Static site generation (SSG)** — Pages are pre-built once at deploy time and served as ready-made HTML.
- **Client-side rendering (CSR)** — The browser receives a nearly empty page and JavaScript builds the UI.
- **Hydration** — JavaScript attaches behavior to server-rendered HTML so it becomes interactive.
- **Hydration mismatch** — The server's HTML and the client's first render disagree, causing flicker or console errors.
- **Streaming** — The server sends HTML in chunks as each part is ready instead of waiting for the whole page.
- **Progressive enhancement** — The page works without JavaScript; scripts layer the richer experience on top.
- **Flash of unstyled content (FOUC)** — Content briefly renders with fallback styles or fonts before the real ones load.
- **Layout shift** — Content jumps around as images, fonts, or data load in, moving what the user was about to click.

### Components & Patterns — structuring UI code
- **Controlled component** — The parent owns an input's value through props and state; every change flows through code.
- **Uncontrolled component** — The DOM owns the input's value; you read it only when you need it.
- **Compound components** — One component split into pieces that share state, like Tabs, Tabs.List, and Tabs.Panel.
- **Composition over configuration** — Build variations by nesting components instead of piling boolean props onto one.
- **Headless component** — Provides behavior and state with no styling, so you bring your own markup.
- **Render prop** — Passing a function for the component to call, letting you decide what it renders.
- **Slot / asChild** — A component renders the element you pass instead of its own wrapper, transferring its behavior onto yours.
- **Portal** — Rendering an element outside its parent's DOM tree, like a modal mounted at the body level.
- **Prop drilling** — Passing data down through layers of components that don't use it, just to reach a deep child.
- **Polymorphic component** — A component that can render as a different tag or component, usually via an `as` prop.

### State & Data — keeping UI and data in sync
- **Optimistic update** — The UI updates immediately as if the request already succeeded, then rolls back if it fails.
- **Single source of truth** — One place owns each piece of state; everything else reads from it instead of keeping copies.
- **Derived state** — A value computed from existing state on the fly instead of stored separately and kept in sync by hand.
- **Lifting state up** — Moving state to the closest shared parent so sibling components stay in sync.
- **Stale-while-revalidate** — Show cached data instantly, then fetch a fresh copy in the background and swap it in.
- **Cache invalidation** — Marking cached data as outdated so the next read fetches fresh data.
- **Race condition** — Two async operations finish in an unexpected order and the stale result overwrites the fresh one.
- **Request waterfall** — Requests that run one after another when they could run in parallel, stacking up their latency.
- **Polling** — Repeatedly asking the server for fresh data on an interval.
- **Idempotent request** — Safe to retry — sending it twice has the same effect as sending it once.
- **Exponential backoff** — Retrying a failed request with growing delays between attempts.

### Events & Input — how the browser reports user actions
- **Debounce** — Wait until the events stop (typing, resizing) before running the handler, so it fires once at the end instead of on every event.
- **Throttle** — Run the handler at most once per interval while events keep firing, like during scrolling.
- **Event bubbling** — An event travels up from the element to its ancestors, so parents hear their children's events.
- **Event delegation** — One listener on a parent handles events from many children instead of a listener on each.
- **Light dismiss** — Closing an overlay by clicking outside it or pressing Escape, rather than an explicit close button.
- **Passive listener** — A scroll or touch listener that promises not to block scrolling, so the browser keeps it smooth.
- **Pointer events** — One event model that covers mouse, touch, and pen instead of separate handlers for each.
- **Hit area** — The region that responds to a tap or click, often padded larger than the visible element.

### CSS Behavior — why styles do what they do
- **Specificity** — The scoring system that decides which of two conflicting CSS rules wins.
- **Cascade** — The order and priority rules CSS uses to resolve competing styles from different sources.
- **Stacking context** — A group whose children's z-index values only compete with each other, not the rest of the page — why a huge z-index can still lose to an element outside the group.
- **Containing block** — The ancestor an absolutely-positioned or percentage-sized element measures itself against.
- **Logical properties** — Start/end instead of left/right, so layouts flip automatically for right-to-left languages.
- **Container query** — Styles that respond to a component's own size instead of the viewport's.
- **Breakpoint** — A viewport width at which the layout changes.
- **Sticky positioning** — An element scrolls normally, then pins in place when it reaches an edge of its container.
- **Scroll snapping** — Scrolling settles onto defined points instead of stopping anywhere, like a carousel.
- **Overscroll behavior** — What happens when scrolling hits a boundary — chaining to the parent, rubber-banding, or nothing.
- **Safe area** — The insets that keep content clear of notches, rounded corners, and the home indicator.

### Performance — making it fast, keeping it fast
- **Code splitting** — Breaking the JavaScript bundle into chunks so each page loads only the code it uses.
- **Lazy loading** — Deferring images, components, or code until they're actually needed.
- **Tree shaking** — Dropping code no one imports from the bundle at build time.
- **Preload** — Telling the browser to fetch a resource now because it's needed very soon, like a font.
- **Prefetch** — Fetching a likely-next resource in idle time, like the page behind a link being hovered.
- **Virtualization** — Rendering only the visible rows of a long list and recycling them as the user scrolls.
- **Memoization** — Caching a computed result so it isn't recomputed when the inputs haven't changed.
- **Re-render** — A component's function runs again to reflect new state or props — cheap alone, expensive in storms.
- **Layout thrashing** — Alternating style reads and writes that force the browser to recalculate layout over and over.
- **Long task** — JavaScript that hogs the main thread, so clicks, typing, and rendering stall until it finishes.
- **Core Web Vitals** — Google's user-experience metrics: LCP (loading), CLS (visual stability), and INP (responsiveness).

### Accessibility Mechanics — the machinery behind an accessible UI
- **Semantic HTML** — Using elements for what they mean (button, nav, h1) so browsers and assistive tech understand the page.
- **ARIA** — Attributes that describe an element's role and state to assistive technology when HTML alone can't.
- **Accessible name** — The text a screen reader announces for an element, from its label, content, or aria-label.
- **Live region** — An area whose content changes are announced by screen readers automatically, like a toast.
- **Focus management** — Deliberately moving keyboard focus — into a dialog when it opens, back to the trigger when it closes.
- **Focus trap** — Keeping Tab cycling inside an open modal instead of escaping to the page behind it.
- **Roving tabindex** — A group is one Tab stop; arrow keys move between the items inside it.
- **inert** — Marking a section unfocusable and invisible to assistive tech, like the page behind a modal.
- **Reduced motion** — Respecting the user's prefers-reduced-motion setting by toning down or removing animation.

### Delivery & the Network — getting assets to the user
- **CDN** — Servers distributed near users that serve static files fast.
- **Cache busting** — Fingerprinting file names with a hash so browsers fetch new versions the moment they change.
- **CORS** — The browser's rules for when a page on one domain may call an API on another.
- **Bundle size** — The total JavaScript shipped to the browser — the budget everything above fights over.
