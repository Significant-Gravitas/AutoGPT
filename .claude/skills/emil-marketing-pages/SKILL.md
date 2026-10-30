---
name: emil-marketing-pages
description: "Design and build marketing surfaces — landing pages, blogs, docs sites, changelogs — with motion restraint, performance, and SEO built in. Use when creating or reviewing a landing page, homepage, hero section, pricing page, blog, documentation site, or changelog; when tempted to add scroll animations, parallax, or carousels; or when setting up fonts, static generation, RSS, or CTAs for a marketing site. Triggers on: landing page, marketing site, marketing page, homepage, hero section, pricing page, blog, blog post, docs, documentation site, changelog, RSS feed, SEO, CTA, call to action, sign up button, get started, scroll animation, fade-in on scroll, parallax, carousel, scroll hijacking, intro animation, page load animation, font preload, layout shift, static generation, SSG, code snippet, copy code button, copy as markdown, nav dropdown, header navigation, waitlist."
---

# Marketing Pages

Marketing pages earn more animation budget than product UI — a visitor sees a landing page once, not a hundred times a day — but the budget is still small, and most sites overspend it. The pages that convert are fast, readable, and honest: content in the DOM, fonts that don't shift, motion that maps to what the user is doing. Restraint is the house style here, from Emil Kowalski's design engineering practice: skip the scroll theatrics, ship the page pre-rendered, and spend your effort on the details visitors feel but never name. When implementing anything below, match the project's existing styling system — Tailwind, plain CSS, CSS Modules, CSS-in-JS — rather than introducing a new one; and match its framework conventions (the examples use Next.js/React, but the principles are framework-agnostic).

## Core Principles

1. **Motion must map to user input.** No scroll-triggered fade-ups, no scroll hijacking, no parallax that isn't 1:1 with scroll, no auto-advancing carousels. If the user didn't cause it, cut it.
2. **Intro animations play once per session.** Gate them with `sessionStorage` so returning within a session skips them, and new sessions see them fresh.
3. **Pre-render everything.** Blog, docs, changelog — build-time generation with revalidation. Never fetch this content at request time.
4. **Kill layout shift at the source.** Preload fonts and above-the-fold images.
5. **Content lives in the DOM.** Hover-revealed nav submenus are visually hidden, not conditionally rendered — crawlers and assistive tech need the real markup.
6. **CTAs know who's looking.** Logged-out visitors get "Get Started"; logged-in users get "Go to Dashboard".
7. **Docs are for copying.** Every code snippet gets a copy button; every page is exportable as markdown; every concept gets a visual example.
8. **Blogs and changelogs are feeds.** Ship RSS, balance headings.

## Motion Restraint

Marketing pages can be more elaborate than product UI, but the same discipline applies.

**Never add:**
- Scroll animations — fade-ups, fade-ins, translate-Y-on-scroll as sections enter the viewport
- Scroll hijacking
- Parallax that doesn't map 1:1 to scroll position
- Auto-advancing carousels

All of these move without the user asking. Motion disconnected from input feels like the page is performing at you.

**Intro animations** (hero reveals, logo sequences) are fine — once. Skip them on subsequent page views within the same session:

```jsx
useEffect(() => {
  const hasSeenIntro = sessionStorage.getItem('hasSeenIntro');
  if (hasSeenIntro) {
    setSkipIntro(true);
  } else {
    sessionStorage.setItem('hasSeenIntro', 'true');
  }
}, []);
```

Use `sessionStorage`, not `localStorage` — the intro should play again on a genuinely new visit, just not on every internal navigation.

## Performance

**Preload fonts** so text doesn't reflow when the webfont lands:

```jsx
import { preload } from 'react-dom';

// In app initialization
preload('/fonts/inter-var.woff2', { as: 'font', type: 'font/woff2' });
```

**Preload above-the-fold images:**

```html
<link rel="preload" as="image" href="/hero-image.webp" />
```

**Statically generate all content pages.** Blog, changelog, docs, and any frequently updated data: build time + revalidation, never request-time fetching:

```jsx
// Next.js example
export async function generateStaticParams() {
  const posts = await getPosts();
  return posts.map((post) => ({ slug: post.slug }));
}

export const revalidate = 3600; // Revalidate every hour
```

## Header Navigation

Submenu content that appears on hover must still exist in the DOM at all times — visually hidden, not mounted-on-hover. This keeps the HTML structure real for SEO and accessibility:

```html
<!-- Content exists in DOM, just visually hidden -->
<nav>
  <button aria-expanded="false">Products</button>
  <div class="submenu" aria-hidden="true">
    <!-- Full content here, not dynamically loaded -->
  </div>
</nav>
```

## Call-to-Action Buttons

CTAs adapt to auth state — a logged-in user being told to "Sign Up" is a dead end:

| State | CTA |
| --- | --- |
| Logged out | "Get Started" or "Sign Up" |
| Logged in | "Go to Dashboard" or "Open App" |

```jsx
<Button href={isLoggedIn ? '/dashboard' : '/signup'}>
  {isLoggedIn ? 'Go to Dashboard' : 'Get Started'}
</Button>
```

## Documentation Sites

- **Copy button on every code snippet.** No exceptions — docs code exists to be pasted.
- **Every page copyable as markdown:** a "Copy as Markdown" button, plus `.md` URLs (e.g. `/docs/getting-started.md` returns raw markdown). This serves both humans and the LLMs increasingly reading your docs.
- **Visual examples everywhere.** Code alone isn't enough — show what the code produces. Let people see and ideally touch the result before they commit to using it.

## Blog & Changelog

- **RSS feeds exist** at predictable paths:

  ```
  /blog/rss.xml
  /changelog/rss.xml
  ```

- **Balance headings** so multi-line titles don't leave a stranded word:

  ```css
  article h1, article h2 {
    text-wrap: balance;
  }
  ```

## Code-Built Illustrations

Decorative illustrations built from DOM elements need three things: an accessible name, no text selection, and no pointer interference:

```jsx
<div
  role="img"
  aria-label="Illustration showing data flow"
  className="illustration"
  style={{
    userSelect: 'none',
    pointerEvents: 'none',
  }}
/>
```

Skip `pointerEvents: 'none'` only if the illustration is interactive.

## Pre-Ship Checklist

- [ ] No scroll-triggered animations, scroll hijacking, non-1:1 parallax, or auto-advancing carousels
- [ ] Intro animation gated behind `sessionStorage`
- [ ] Fonts preloaded; no layout shift on load
- [ ] Above-the-fold images preloaded
- [ ] Blog/docs/changelog statically generated with revalidation
- [ ] Nav submenu content present in the DOM when closed
- [ ] CTAs switch copy and destination based on auth state
- [ ] Docs snippets have copy buttons; pages export as markdown (`.md` URLs)
- [ ] Docs show visual examples, not just code
- [ ] RSS feeds live for blog and changelog
- [ ] `text-wrap: balance` on article headings
- [ ] Code-built illustrations have `aria-label`, disabled selection, disabled pointer events
