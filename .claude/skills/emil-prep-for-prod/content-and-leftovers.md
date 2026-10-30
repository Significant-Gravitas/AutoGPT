# Lane H — Content & Leftovers

Two halves. **Content** is the text and type a visitor reads: whether it's real, whether it renders the way it was designed, whether it survives real data. **Leftovers** are what the build process left behind: debug output, dev tools, scaffolding routes, and the tells an agent leaves in a diff. Owning skills: `emil-typography` and `emil-design-foundations` for content, `emil-unslop-code` and `emil-unslop-design` for leftovers.

This lane finds more blockers per minute than any other, because nothing in it shows up as a bug. The app works. It just says "Lorem ipsum" on the pricing page and logs the session token to the console.

Hard Rule 8 governs the content half: you can delete a fake testimonial, but you can't write a real one. Missing real content goes on the human list.

## Content

### H1. No placeholder content — BLOCKER on a public page

**Hunt for** (case-insensitive, across `src` and content directories):

```
lorem|ipsum|dolor sit          placeholder text|your text here
John Doe|Jane Doe|Acme         example\.com|test@|foo@bar
TODO|TBD|coming soon|WIP       Untitled|New Page|My App
\$0\.00|\$99|XX|###            via\.placeholder|placehold\.co|picsum|unsplash\.it
```

Also: the framework's default favicon and `<title>` ("Create Next App", "Vite + React"); `href="#"` and `href=""` on links that were never wired up (footer columns and social icons are the usual place); image `alt` text that is the filename.

A seeded demo in a docs example is fine. Vet each hit.

### H2. No fabricated proof — BLOCKER

Agents generating a landing page invent the social proof to fill the template: "Trusted by 10,000+ teams", a five-star testimonial from "Sarah K., Product Lead", a logo wall of companies that aren't customers, "99.9% uptime", "SOC 2 compliant". On a live site these are false claims, and the compliance ones are legal exposure.

**Hunt for:** testimonial, stats, logo-wall, and trust-badge sections. For each number, quote, and logo, find its source in the repo (a CMS entry, a data file, a commit by a human). No source means it's a finding.

Never fix this by rewording. Remove the section or flag it; the user confirms which claims are real.

### H3. Buttons and errors say something specific — SHOULD FIX

"Submit", "OK", and "Click here" make the user guess the outcome. "Something went wrong" and "Invalid input" make them guess the fix.

**Hunt for:** button labels `Submit` / `OK` / `Click here` / `Yes` / `No`; error strings `Something went wrong` / `An error occurred` / `Invalid` / `Error`; links whose whole text is "here" or "learn more" (also unreadable out of context to a screen reader).

| Instead of | Write |
| --- | --- |
| "Submit" | "Save changes", "Send invite", "Create project" |
| "Invalid input" | "Your email must include an @" |
| "No items" | "No projects yet. Create one to get started." |
| "Please confirm your selection before proceeding" | "Confirm" |

Sentence case for UI, not Title Case On Every Label. When the right label is obvious from the handler (`onClick={saveChanges}`), apply it. When it isn't, flag it.

### H4. Missing weights fail visibly — SHOULD FIX

When CSS asks for a weight or italic that was never loaded, the browser fakes it: smeared bold, slanted roman. It looks subtly wrong on every heading and nobody can say why. `font-synthesis: none` turns the fake into an obvious fallback, so the gap gets noticed and fixed.

**Hunt for:** `font-weight` values used in CSS vs the weights loaded in `@font-face` / `next/font`; `italic` / `<em>` with no italic file loaded; no `font-synthesis` anywhere.

```css
html { font-synthesis: none; }
```

Apply it, then list any weight that's used but not loaded. Loading a new font file is the user's call (it costs bytes, see B2); using the nearest loaded weight is the other option.

### H5. Headings and paragraphs break well — POLISH

A headline that wraps to "seven words on the first line / one" looks unfinished at exactly the breakpoint nobody checked.

```css
h1, h2, h3 { text-wrap: balance; }
p { text-wrap: pretty; }
```

Body copy caps near `65ch`. A paragraph running the full width of a 1440px screen is work to read.

### H6. Real typographic characters — POLISH

`…` not `...`, curly quotes and apostrophes (`’` `“` `”`) not straight ones in rendered copy, `×` not `x` for dimensions, `–` for ranges. Code, identifiers, and anything inside `<code>` stay straight.

**Hunt for:** `...` in JSX text and string literals shown to users ("Loading...", "Saving..."); straight `'` in contractions in marketing copy. Don't touch strings that are keys, test IDs, or log messages.

### H7. Casing lives in CSS — POLISH

`<span>GET STARTED</span>` is shouted by screen readers on some setups, can't be restyled, and breaks search and copy-paste. Write sentence case in the markup and apply `text-transform: uppercase` with slight positive tracking.

### H8. The locale is declared — SHOULD FIX

`<html>` with no `lang` breaks hyphenation, quote rendering, screen reader pronunciation, and browser translation prompts. One attribute (also A12).

Dates, numbers, and currency formatted by hand (`${month}/${day}`, `"$" + price.toFixed(2)`) are wrong for most of the world. Use `Intl.DateTimeFormat` / `Intl.NumberFormat` when the app serves more than one locale; note it when it doesn't.

## Leftovers

### H9. Dev tools aren't mounted in production — BLOCKER

An animation inspector, a control panel, or a devtools drawer visible to every visitor. It also ships its bytes to every visitor (B18).

**Hunt for:** `<Lapse`, `mountLapse(`, `@aiforui/lapse/install`; `leva` / `<Leva` / `useControls`; `ReactQueryDevtools`; `tweakpane`, `dat.gui`, `stats.js`; `react-scan`, `why-did-you-render`; `eruda` / `vConsole`; debug overlays and grid overlays toggled by a constant; feature-flag panels. For each, find the mount and check it's gated.

```jsx
{process.env.NODE_ENV === "development" && <Lapse />}
```

```js
if (import.meta.env.DEV) {
  const { mountLapse } = await import("@aiforui/lapse/panel");
  mountLapse();
}
```

A dynamic import behind the gate keeps the package out of the production bundle; a static import with a gated render still ships the bytes. `<Leva hidden>` hides the panel but `useControls` values still drive the UI, so check that the shipped defaults are the approved ones. When a control panel was used to dial something in (`emil-prototype`, `emil-build-a-tool`), the approved values live in a saved config file that production imports, never in a panel default.

### H10. Scaffolding routes aren't publicly reachable — BLOCKER

`/proto/*`, `/lab/*`, `/test`, `/playground`, `/sandbox`, `/debug`, `/styleguide`, preview routes added to dodge auth. They're indexed, they often bypass auth, and they show unfinished work.

**Hunt for:** route directories with those names; pages imported by nothing and linked from nowhere; a `sitemap` that lists them.

Hard Rule 9: finding them is the job; deleting them is not. `emil-prototype` directories are throwaway by design and the user usually wants them gone; `emil-build-a-tool` leaves tools in place on purpose. Gate them out of production and let the user decide:

```ts
// in the route
if (process.env.NODE_ENV === "production") notFound();
```

Add them to `robots` disallow as a second layer, never as the only one.

### H11. No debug output — BLOCKER when it prints user data, SHOULD FIX otherwise

**Hunt for:** `console.log` / `console.debug` / `console.table` / `console.dir`; `debugger`; `alert(`; `JSON.stringify(` rendered into the page inside a `<pre>`.

Vet each: `console.error` in an error boundary or a `catch` is logging, keep it. `console.log(user)`, `console.log(session)`, `console.log(res)` on an auth or payment response is a leak; remove it and say so in the report. If the project has a logger, route what's worth keeping through it.

### H12. No secrets or dev URLs in client code — BLOCKER

Anything in a client bundle is public. A secret key behind `NEXT_PUBLIC_` / `VITE_` is published. A hardcoded `http://localhost:3000` works on one machine.

**Hunt for:** `localhost`, `127.0.0.1`, `ngrok`, staging hostnames in source (not config); `NEXT_PUBLIC_*` / `VITE_*` / `PUBLIC_*` variable names containing `SECRET`, `PRIVATE`, `SERVICE_ROLE`, or `_KEY` where the service's public key isn't meant; string literals shaped like keys (`sk_live_`, `sk_test_`, `eyJ…`); `.env` files tracked by git.

Report the file and the variable *name*, never the value. Rotating a leaked key is the user's job and it's urgent; say so at the top of the report, above the verdict.

### H13. Commented-out code and stale markers — SHOULD FIX

**Hunt for:** blocks of commented-out JSX or logic; `TODO` / `FIXME` / `HACK` / `XXX`; `@ts-ignore` / `@ts-expect-error` / `as any` / `eslint-disable` added recently; `.only(` / `.skip(` in tests; `V2` / `New` / `Old` / `Copy` / `Temp` in component and file names.

Commented-out code gets deleted; git remembers it. A `TODO` is triaged, not deleted: read it, and if it describes something that isn't done ("TODO: check permissions here"), it's a finding in its own right and probably the most important one in the lane. Report every `TODO` on a security, auth, payment, or data-deletion path verbatim.

### H14. Agent tells in the diff — POLISH

Comments that narrate the code (`// Set the state to loading`), comments that talk to a reviewer (`// Updated per your feedback`), `try/catch` that logs and continues around code that can't fail, one-call wrapper functions, duplicate helpers, unused imports and exports. None of them break anything. All of them make the next bug harder to find. Run `/emil-unslop-code` on the diff after launch; during the ship check, remove only the ones you're already touching.

### H15. Generated-looking surfaces — POLISH

The indigo gradient hero, gradient headline text, glass cards over nothing, a three-card grid of emoji features, sparkle icons meaning "AI", a fade-in on every section. These don't break anything either, and the week before launch is the wrong time to redesign (Operating Posture, failure mode 3). Note them in one line and name `/emil-unslop-design` for after launch. Emoji used as functional icons in product UI are the exception: they render differently on every platform, so flag them as SHOULD FIX.

### H16. Read every public page out loud — always

Put on the human list: "read every word on the landing, pricing, and signup pages", "click every link in the header and footer (placeholder `#` hrefs and links to `/` are common)", "check that every number, logo, and quote on the site is real". No tool verifies truth. Report these as checks to run, never as verified.
